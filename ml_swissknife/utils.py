"""All-purpose utility file with helpers for

- PyTorch tensor processing (flattening, unflattening, vjp, jvp),
- plotting,
- custom gradient checking (gradient wrt parameters, and second-order gradients),
- meters (ema, online average),
- custom learning rate schedules,
- ema model averaging,
- google cloud storage utilities,
- custom context managers (Timer, DisableGC),
- checkpoint storage/loading, and
- data loaders,
- misc log sanitization.
"""

import abc
import argparse
import collections
import contextlib
import copy
import csv
import datetime
import functools
import gc
import io
import json
import logging
import math
import os
import pathlib
import random
import shutil
import signal
import sys
import time
import warnings
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import requests
import six
import torch
import torch.nn.functional as F
import tqdm
from scipy import stats
from torch import nn, optim
from torch.utils import data

from .types import Numeric, PathOrIOBase

# Misc.
home = os.path.expanduser("~")
home_data = os.path.join(home, "data")
join = os.path.join
pathexists = os.path.exists
makedirs = functools.partial(os.makedirs, exist_ok=True)
dirname = os.path.dirname
basename = os.path.basename


def float2str(x, precision=8):
    return f"{x:.{precision}f}".replace(".", "_")


def int2str(x, leading_zeros=8):
    return f"{x:0{leading_zeros}d}"


def dict2str(d: dict):
    """Return a pretty string version of the dictionary.

    Useful for annotating runs based on hparams and configs.
    """
    output_str = ""
    for index, (key, val) in enumerate(d.items(), 1):
        if index < len(d):
            output_str = output_str + f"{key}_{val}-"
        else:
            output_str = output_str + f"{key}_{val}"
    return output_str


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def average_over_seed(seq_of_seq):  # Here purely due to backward compatibility.
    min_len = min(len(seq) for seq in seq_of_seq)
    seq_of_seq = [seq[:min_len] for seq in seq_of_seq]
    seq_of_seq = np.array(seq_of_seq)
    return seq_of_seq.mean(0), seq_of_seq.std(0)


def average_metric_over_seeds(
    *seqs: Union[Sequence[Numeric], Sequence[Dict[str, Numeric]]],
):
    # seqs is an iterable. Each seq is a sequence of numbers or dicts to average over.
    single_input = len(seqs) == 1
    outputs = tuple(_average_metric_over_seeds_for_single_seq(seq) for seq in seqs)
    if single_input:
        return outputs[0]
    return outputs


def _average_metric_over_seeds_for_single_seq(
    seq: Union[Sequence[Numeric], Sequence[Dict[str, Numeric]]],
):
    # TODO: Enable further nesting, e.g., dict where values could be list or tuple.
    if len(seq) == 0:
        return ()
    if isinstance(seq[0], (tuple, list)):
        # Returns the mean and std.
        return float(np.mean(seq)), float(np.std(seq))
    elif isinstance(seq[0], dict):
        # We assume each dict has the same set of keys.
        # Returns a dict, each key maps to a tuple of mean and std.
        keys = seq[0].keys()
        output = dict()
        for key in keys:
            values = [seq_i[key] for seq_i in seq]
            output[key] = (float(np.mean(values)), float(np.std(values)))
        return output
    else:
        raise ValueError(
            f"Expected each elem of seq to be of type int, float, or dict, but found type: {type(seq[0])}"
        )


def single_standard_deviation(sample, return_type="tuple"):
    if return_type == "tuple":
        return np.mean(sample), np.std(sample)
    elif return_type == "dict":
        return dict(mean=np.mean(sample), delta=np.std(sample))
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def confidence_interval(sample, alpha=0.05):
    """Compute (asymptotic) confidence interval under the normality assumption.

    Assumes each sample is drawn from a normal distribution.
    This could still work if you have a large number of samples.
    """
    alpha2zscore = {
        0.01: 2.58,
        0.02: 2.33,
        0.05: 1.960,
        0.1: 1.645,
        0.15: 1.440,
    }

    if isinstance(sample, (list, tuple)):
        sample = torch.tensor(sample)
    sample: torch.Tensor
    if not sample.dim() == 1:
        raise ValueError(f"`sample` must be 1-dimensional.")
    sample_size = len(sample)
    sample_mean = sample.mean()
    sample_std = sample.std(unbiased=False)
    zscore = alpha2zscore[alpha]

    delta = zscore * sample_std / math.sqrt(sample_size)
    low = sample_mean - delta
    high = sample_mean + delta
    return dict(low=low, high=high, delta=delta, mean=sample_mean)


def _make_w_io_base(f: PathOrIOBase, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            makedirs(f_dirname)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f: PathOrIOBase, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(
    obj: Union[str, dict, list], f: PathOrIOBase, mode="w", indent=4, default=str
):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f: PathOrIOBase, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdumps(obj, indent=4, default=str):
    return json.dumps(obj, indent=indent, default=default)


jloads = json.loads


def jldump(seq: Sequence[dict], f: PathOrIOBase, mode="w", indent=4, default=str):
    """Dump a sequence of dictionaries into a .jsonl file."""
    if not all(isinstance(item, dict) for item in seq):
        raise ValueError("Input is not of type Sequence[dict].")
    f = _make_w_io_base(f, mode)
    for item in seq:
        f.write(json.dumps(item, indent=indent, default=default) + "\n")
    f.close()


def jlload(f: PathOrIOBase, mode="r", strip=True):
    """Load a .jsonl file into a list of dictionaries."""
    return [json.loads(line) for line in readlines(f, mode=mode, strip=strip)]


def read_csv(f: PathOrIOBase, mode="r", delimiter="\t"):
    f = _make_r_io_base(f, mode)
    reader = csv.DictReader(f, delimiter=delimiter)
    out = dict(fieldnames=reader.fieldnames, rows=tuple(line for line in reader))
    f.close()
    return out


def write_csv(
    f: PathOrIOBase,
    fieldnames: Union[List, Tuple],
    # Each entry in rows is either a sequence or a dict.
    rows: Sequence[Union[Sequence, Dict]],
    mode="w",
    delimiter="\t",
):
    f = _make_w_io_base(f, mode)
    if isinstance(rows[0], dict):
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    else:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    f.close()


def readlines(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    f = _make_r_io_base(f, mode)
    lines = f.readlines()
    if strip:
        lines = [line.strip() for line in lines]
    f.close()
    return lines


def read(f: Union[str, pathlib.Path, io.IOBase], mode="r", strip=True):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    content = f.read()
    if strip:
        content = content.strip()
    f.close()
    return content


def listdir(
    directory,
    skip_suffixes: Union[Sequence, str] = (),
    full_path: Optional[bool] = False,
):
    """Convenience function to replace `os.listdir` for skipping annoying mac hidden files."""
    if isinstance(skip_suffixes, str):
        skip_suffixes = (skip_suffixes,)
    skip_suffixes = tuple(skip_suffixes) + (".DS_Store",)

    file_names = os.listdir(directory)
    for skip_suffix in skip_suffixes:
        file_names = [
            file_name for file_name in file_names if not file_name.endswith(skip_suffix)
        ]

    if full_path:
        file_names = [os.path.join(directory, file_name) for file_name in file_names]
    return file_names


def list_file_paths(directory, skip_suffixes: Union[Sequence, str] = ()):
    """Recursively go down the folder, list only files, and returns paths."""
    if isinstance(skip_suffixes, str):
        skip_suffixes = (skip_suffixes,)
    skip_suffixes = tuple(skip_suffixes) + (".DS_Store",)

    file_paths = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(directory)
        for file in files
    ]
    for suffix in skip_suffixes:
        file_paths = [
            file_path for file_path in file_paths if not file_path.endswith(suffix)
        ]
    return file_paths


# Backwards compat.
listfiles = list_file_paths


def listfds():
    """List all open file descriptors."""
    return sorted(list(set(os.listdir("/proc/self/fd/"))))


def compress(path: str, out_path: Optional[str] = None):
    """Compress a file or folder; relies on `pigz`, a Linux utility.

    Args:
        path: Path to the file or folder to compress.
        out_path: Path to the compressed file; defaults to the original path with the suffix `tar.gz` added.

    Returns:
        None.
    """
    if out_path is None:
        out_path = path + ".tar.gz"

    with Timer(msg=f"Compressed file/folder: {path}", logging=True):
        os.system(f"tar cf - {path} | pigz -9 > {out_path}")


def decompress(path: str, out_path: Optional[str] = None):
    """Decompress a file or a folder; relies on `pigz`, a Linux utility with multi-threading.

    Args:
        path (str): Path to file/folder to be decompressed.
        out_path (str): Path to folder to put the decompressed, defaults to `None` which is current directory.
    """
    with Timer(msg=f"Decompressed file: {path}", logging=True):
        if out_path is not None:
            os.system(f"tar -I pigz -xf {path} -C {out_path}")
        else:
            os.system(f"tar -I pigz -xf {path}")


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.
    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.
    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def write_config(
    args: argparse.Namespace,
    file_name="argparse.json",
    config_path=None,
    attr="train_dir",
):
    """Creates folders and write config.

    Doesn't write if in `eval` mode.
    """
    if not hasattr(args, attr):
        return

    train_dir = getattr(args, attr)
    if train_dir is None:
        return

    os.makedirs(train_dir, exist_ok=True)
    if config_path is None:
        config_path = os.path.join(train_dir, file_name)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(args.__dict__, f, indent=4)
    logging.warning(f"Wrote config: {config_path}")

    if (hasattr(args, "cloud_storage") and args.cloud_storage) or (
        hasattr(args, "to_gcs") and args.to_gcs
    ):
        gs_upload_from_path(config_path, remove_local=False)
        logging.warning(f"Uploaded to cloud: {config_path}")


def load_config(
    args: argparse.Namespace,
    file_name="argparse.json",
    config_path=None,
    replace_exclude=(),
):
    if config_path is None:
        config_path = os.path.join(args.train_dir, file_name)
    with open(config_path, "r") as f:
        config = json.load(f)
    for key in replace_exclude:
        config.pop(key, None)
    args.__dict__ = {**args.__dict__, **config}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def str2int(v):
    if isinstance(v, int):
        return v
    if v.lower() in ("none",):
        return None
    return int(v)


def gather_args(parser: argparse.ArgumentParser):
    """Gathers known and unknown args together.

    Unknown args are arguments whose names we don't known before hand, and they aren't specified by `add_argument`.
    """
    args, unknown_args = parser.parse_known_args()
    unknown_options = collections.defaultdict(list)

    key = None
    for arg in unknown_args:
        if arg.startswith("--"):
            key = arg[2:]
        elif arg.startswith("-"):
            key = arg[1:]
        else:
            unknown_options[key].append(arg)
    args.__dict__ = {**args.__dict__, **unknown_options}
    return args


def flatten_nested_pystruct(sequence: Sequence):
    """Flatten nested python list/tuple/set and return a list of elements."""
    if not isinstance(sequence, (tuple, list, set)):
        return [sequence]
    return [i for entry in sequence for i in flatten_nested_pystruct(entry)]


def parallel_sort(*args, key=None, reverse=False):
    """Parallel sort of multiple lists."""
    # args: A bunch of sequences.
    if key is None:
        key = lambda inputs: inputs[
            0
        ]  # Parallel sort based on the order of the first list.
    ret = sorted(zip_(*args), key=key, reverse=reverse)
    return tuple([ret_i[j] for ret_i in ret] for j in range(len(args)))


def linregress_slope(x, y):
    """Return the slope of a least-squares regression for two sets of measurements."""
    return stats.linregress(x, y)[0]


def pretty_str(names: Sequence, vars: Sequence, precision: Optional[float] = 4):
    ret = ""
    for name, var in zip(names[:-1], vars[:-1]):
        if isinstance(var, float):
            ret += f"{name}: {var:.{precision}f}, "
        else:
            ret += f"{name}: {var}, "
    ret += f"{names[-1]}: {vars[-1]}"  # No comma after last.
    return ret


class _SuppressAssertions(object):
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write("Caught AssertionError: " + str(exc_val))
            return True


def show_env(args_or_device=None):
    if args_or_device is not None:
        if hasattr(args_or_device, "device"):
            args_or_device = args_or_device.device
        elif hasattr(args_or_device, "no_gpu"):
            args_or_device = (
                "cuda"
                if torch.cuda.is_available() and not args_or_device.no_gpu
                else "cpu"
            )
        logging.warning(f"Running on device: {args_or_device}")
    logging.warning(f"CUDA device count: {torch.cuda.device_count()}")
    logging.warning(
        f"Running Python: \n{sys.version}; \nversion info: \n{sys.version_info}"
    )
    logging.warning(f"Running PyTorch: {torch.__version__}")
    logging.warning(f"Running six: {six.__version__}")


def download_file_from_google_drive(id, destination, timeout=120):
    """Download a file hosted on Google drive with the id extracted from a sharable link."""

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in tqdm.tqdm(response.iter_content(CHUNK_SIZE), desc="chunks"):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id}, stream=True, timeout=timeout)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True, timeout=timeout)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    save_response_content(response, destination)


def isdigit(x):
    if len(x) > 0 and x[0] in ("-", "+"):
        return x[1:].isdigit()
    return x.isdigit()


class Timeout(contextlib.ContextDecorator):
    def __init__(self, seconds: Union[float, int], error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)  # This doesn't work in Py2.

    def __enter__(self):
        signal.signal(
            signal.SIGALRM, self.handle_timeout
        )  # Create custom handler for SIGALRM.
        # `signal.ITIMER_REAL` will deliver SIGALRM in `self.seconds`.
        # Better than directly sending SIGALRM, which doesn't allow floating-point seconds.
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # Cancel the previously set alarm if haven't timed out.


def rm(*paths: str):
    """Remove path or directory specified at path."""
    for path in paths:
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def deduplicate(x: Union[List, Tuple]):
    """Remove duplicates in a list or tuple; preserves original order."""
    return type(x)(dict.fromkeys(x))


def mvdir(src: str, tgt: str, tmp: str):
    """Move source directory to target directory.

    Most helpful when you want to insert subdirectory in path, e.g.,
    moving /home/path -> /home/path/sub.
    Note naive `mv` does not work for this case!
    """
    os.system(f"mv {src} {tmp}")
    os.system(f"mkdir -p {tgt}")
    os.system(f"mv {tmp}/* {tgt}")
    os.system(f"rm -r {tmp}")


def handle_unused_kwargs(unused_kwargs, msg=None):
    if len(unused_kwargs) > 0:
        if msg is not None:
            warnings.warn(f"{msg}: Unexpected arguments {unused_kwargs}")
        else:
            warnings.warn(f"Unexpected arguments {unused_kwargs}")


class ContainerMeta(type):
    def all(cls):
        return sorted(getattr(cls, x) for x in dir(cls) if not x.startswith("__"))

    def __str__(cls):
        return str(cls.all())

    def __contains__(cls, item):
        return item in cls.all()


def run_tasks(
    task: str,
    task_names: Sequence[str],
    task_callables: Sequence[Callable],
    **kwargs,  # Given to each callable.
):
    for task_name, task_callable in zip_(task_names, task_callables):
        if task == task_name:
            return task_callable(**kwargs)
    raise ValueError(f"Unknown task: {task}. Expected one of {task_names}")


def runs_tasks(*args, **kwargs):
    logging.warning(
        "`runs_tasks` will be deprecated in the future. Consider using `run_tasks` instead."
    )
    return run_tasks(*args, **kwargs)


# Torch.
def tsave(state_dicts: dict, path: str):
    makedirs(dirname(path), exist_ok=True)
    torch.save(state_dicts, path)


def collect_tensors(verbose=False):
    """Collect all referenced tensors; useful for debugging memory leak."""
    count = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                if verbose:
                    print(type(obj), obj.size())
            count += 1
        except Exception:
            pass
    logging.warning(f"Total number of tensors: {count}")


def log_shape(*args):
    for i, arg in enumerate(args):
        logging.warning(f"tensor {i}, shape: {arg.shape}")


def flatten(possibly_sequence: Union[torch.Tensor, Sequence[torch.Tensor]]):
    if torch.is_tensor(possibly_sequence):
        return possibly_sequence.reshape(-1)
    return (
        torch.cat([p.reshape(-1) for p in possibly_sequence])
        if len(possibly_sequence) > 0
        else torch.tensor([])
    )


def flatten_nested(possibly_sequence: Union[torch.Tensor, Sequence]):
    if torch.is_tensor(possibly_sequence):
        return possibly_sequence.reshape(-1)
    flat_tensors = [flatten_nested(entry) for entry in possibly_sequence]
    return torch.cat(flat_tensors, dim=0) if len(flat_tensors) > 0 else torch.tensor([])


def ravel_pytree(
    possibly_sequence: Union[Sequence, torch.Tensor],
) -> Tuple[torch.Tensor, Callable]:
    if torch.is_tensor(possibly_sequence):
        return possibly_sequence.reshape(-1), lambda x: x.reshape(
            possibly_sequence.size()
        )

    def make_unravel(size):  # Need this function to copy size!
        return lambda x: x.reshape(size)

    unravels, flats, numels = [], [], []
    for entry in possibly_sequence:
        if torch.is_tensor(entry):
            unravel_i = make_unravel(entry.size())
            flat_i = entry.reshape(-1)
        else:
            flat_i, unravel_i = ravel_pytree(entry)
        unravels.append(unravel_i)
        flats.append(flat_i)
        numels.append(flat_i.numel())

    def unravel(flat: torch.Tensor):
        return [
            unravel_(flat_)
            for flat_, unravel_ in zip_(flat.split(split_size=numels), unravels)
        ]

    return torch.cat(flats) if len(flats) > 0 else torch.tensor([]), unravel


def unnest_pytree(possibly_nested: Union[Sequence, Any]) -> Tuple[List, Callable]:
    if not isinstance(possibly_nested, (tuple, list)):
        return [possibly_nested], lambda x: x[0]

    flats: List[List] = []
    nests: List[Callable] = []
    numels: List[int] = []
    for entry in possibly_nested:
        flat_i, nest_i = unnest_pytree(entry)
        flats.append(flat_i)
        nests.append(nest_i)
        numels.append(len(flat_i))
    flat = [elem for flat_i in flats for elem in flat_i]

    def nest(flat):
        original = []
        cumsum = np.cumsum([0] + numels)
        for start, end, nest in zip_(cumsum[:-1], cumsum[1:], nests):
            flat_i = flat[start:end]
            original.append(nest(flat_i))
        return original

    return flat, nest


def cat(args: Union[Tuple, List], dim=0, out=None):
    """Concatenation with broadcasting."""
    size = [max(dims) for dims in zip_(*[list(t.size()) for t in args])]
    return torch.cat([t.expand(size) for t in args], dim=dim, out=out)


def fill_tail_dims(y: torch.Tensor, y_like: torch.Tensor):
    """Fill in missing trailing dimensions for y according to y_like."""
    return y[(...,) + (None,) * (y_like.dim() - y.dim())]


def channel_cat(t, y):
    t = fill_tail_dims(t, y).expand_as(y[:, :1, ...])
    return torch.cat((t, y), dim=1)


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return _swish(x)


@torch.jit.script
def _swish(x):
    return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return _mish(x)


@torch.jit.script
def _mish(x):
    return x * torch.tanh(F.softplus(x))


def flat_to_shape(flat_tensor: torch.Tensor, shapes: Sequence):
    """Convert a flat tensor to a list of tensors with specified shapes.

    `flat_tensor` must have exactly the number of elements as stated in `shapes`.
    """
    numels = [shape.numel() for shape in shapes]
    return [
        flat.reshape(shape)
        for flat, shape in zip_(flat_tensor.split(split_size=numels), shapes)
    ]


def convert_none_to_zeros(
    sequence: Sequence[Union[torch.Tensor, type(None)]],
    like_sequence: Sequence[torch.Tensor],
):
    return [
        torch.zeros_like(q) if p is None else p for p, q in zip(sequence, like_sequence)
    ]


def make_seq_requires_grad(sequence: Sequence[torch.Tensor]):
    return [p if p.requires_grad else p.detach().requires_grad_(True) for p in sequence]


def is_strictly_increasing(ts):
    return all(x < y for x, y in zip(ts[:-1], ts[1:]))


def make_any_check(func):
    def any_check(*args, **kwargs):
        inps = [arg for arg in args] + list(kwargs.values())
        return any(func(inp) for inp in inps)

    return any_check


isnan = make_any_check(lambda t: torch.isnan(t).any())
isinf = make_any_check(lambda t: torch.isinf(t).any())


def isnan_or_isinf(*args, **kwargs):
    return isnan(*args, **kwargs) or isinf(*args, **kwargs)


def vjp(outputs, inputs, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [
        torch.as_strided(i, (), ()) for i in inputs
    ]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    _vjp = torch.autograd.grad(outputs, inputs, **kwargs)
    return convert_none_to_zeros(_vjp, inputs)


def jvp(outputs, inputs, grad_inputs=None, **kwargs):
    if torch.is_tensor(inputs):
        inputs = [inputs]
    _dummy_inputs = [
        torch.as_strided(i, (), ()) for i in inputs
    ]  # Workaround for PyTorch bug #39784.

    if torch.is_tensor(outputs):
        outputs = [outputs]
    outputs = make_seq_requires_grad(outputs)

    dummy_outputs = [torch.zeros_like(o, requires_grad=True) for o in outputs]
    first_kwargs = copy.deepcopy(kwargs)
    first_kwargs["create_graph"] = True
    _vjp = torch.autograd.grad(
        outputs, inputs, grad_outputs=dummy_outputs, **first_kwargs
    )  # Must create graph to backprop a second time.
    _jvp = torch.autograd.grad(_vjp, dummy_outputs, grad_outputs=grad_inputs, **kwargs)
    return convert_none_to_zeros(_jvp, dummy_outputs)


def to_numpy(*possibly_tensors: Union[torch.Tensor, np.ndarray, float]):
    arrays = possibly_tensors
    arrays = [
        t.item() if isinstance(t, torch.Tensor) and t.numel() == 1 else t
        for t in arrays
    ]
    arrays = [
        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in arrays
    ]
    return arrays[0] if len(arrays) == 1 else arrays


def manual_seed(
    args_or_seed: Union[int, argparse.Namespace], hardcore=True, disable_tf=True
):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    if hardcore:
        # Avoid letting cudnn heuristics affect results.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if not disable_tf:
        try:
            import tensorflow as tf

            tf.random.set_seed(args_or_seed)
        except ModuleNotFoundError:
            logging.warning("Tensorflow not installed; ignoring set seed for tf.")


@contextlib.contextmanager
def tmp_seed(
    args_or_seed: Optional[Union[int, argparse.Namespace]],
    is_cuda: bool = torch.cuda.is_available(),
):
    """Context manager to use a temporary random seed with `with` statement."""

    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random_state = random.getstate()
    if is_cuda:
        torch_cuda_state = torch.cuda.get_rng_state()

    if args_or_seed is None:
        # if you want to allow hardcore you need to restore in the finally
        manual_seed(args_or_seed, hardcore=False, disable_tf=True)

    try:
        yield
    finally:
        if seed is not None:
            # if seed is None do as if no tmp_seed
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            if is_cuda:
                torch.cuda.set_rng_state(torch_cuda_state)


def convert_str_dtype_to_torch_dtype(str_dtype: Optional[str]):
    if str_dtype in ("single", "float32", "float", "fp32", None):
        return torch.float
    elif str_dtype in ("half", "float16", "fp16"):
        return torch.float16
    elif str_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif str_dtype in ("double", "float64"):
        return torch.float64
    else:
        raise ValueError(f"Unknown dtype: {str_dtype}")


def manual_dtype(args_or_dtype: Union[str, argparse.Namespace]):
    dtype = args_or_dtype.dtype if hasattr(args_or_dtype, "dtype") else args_or_dtype
    if dtype in ("float64", "double"):
        torch.set_default_dtype(torch.float64)
    elif dtype in ("float16", "half"):
        torch.set_default_dtype(torch.float16)


def trainable_parameters(*modules: nn.Module):
    """Return the parameters which require gradient."""
    single_module = len(modules) == 1
    outs = [
        [param for param in module.parameters() if param.requires_grad]
        for module in modules
    ]
    if single_module:
        return outs[0]
    return outs


def count_parameters(*modules: nn.Module, only_differentiable: Optional[bool] = False):
    """Count the number of parameters for each module."""
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [
            [p for p in param_list if p.requires_grad] for param_list in param_lists
        ]
    numels = [sum(p.numel() for p in param_list) for param_list in param_lists]
    return numels[0] if len(modules) == 1 else numels


def count_parameter_tensors(
    *modules: nn.Module, only_differentiable: Optional[bool] = False
):
    param_lists = [list(m.parameters()) for m in modules]
    if only_differentiable:
        param_lists = [
            [p for p in param_list if p.requires_grad] for param_list in param_lists
        ]
    lens = [len(param_list) for param_list in param_lists]
    return lens[0] if len(modules) == 1 else lens


def count_tensor_list_size(
    tensor_list: Union[torch.Tensor, Sequence, Iterator], format="byte"
):
    """Approximately count the size of a list of tensors in terms of bytes."""
    if torch.is_tensor(tensor_list):
        tensor_list = [tensor_list]
    _bytes = 4 * sum([t.numel() for t in tensor_list])
    if format == "byte":
        return _bytes
    elif format == "kb":
        return _bytes / 1024
    elif format == "mb":
        return _bytes / 1024**2
    elif format == "gb":
        return _bytes / 1024**3
    else:
        raise ValueError(f"Unknown format: {format}")


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = tuple(shape)

    def forward(self, x):
        return x.view((-1,) + self.shape)


class FuncWrapper(nn.Module):
    def __init__(self, func):
        super(FuncWrapper, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class NormalizedEmbedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(NormalizedEmbedding, self).__init__(*args, **kwargs)
        nn.init.normal_(self.weight, std=self.embedding_dim**-0.5)


def subsequent_mask(size, device=None):
    """Mask out subsequent positions.

    Useful for transformer training.
    """
    return torch.triu(torch.ones((1, size, size), device=device), diagonal=1) == 0


def masks_from_lengths(lengths: torch.Tensor):
    """Create True/False mask based on lengths.

    Useful for masking out padding tokens.
    """
    return torch.arange(max(lengths), device=lengths.device)[None, :] < lengths[:, None]


def evaluate_prettiness(
    sampler=None,
    folder=None,
    input_2="cifar10-train",
    n=50000,
    batch_size=1000,
    clean_afterwards=False,
    fid=False,
    isc=False,
    kid=False,
):
    """Evaluate a generative model in terms of IS, FID, or KID.

    At least one of `model` or `folder` must be present.

    Args:
        sampler (object, optional): An objective with the method `func` that samples from the model.
        folder (str, optional): Path to the folder that contains all the images.
        input_2 (str, optional): Name of registered dataset or a path to a folder.
        n (int, optional): Number of samples to take.
        batch_size (int, optional): Number of samples in each batch.
        clean_afterwards (bool, optional): Clean the local cache if True.

    Returns:
        A dictionary of metric values.
    """
    import matplotlib.pyplot as plt
    import torch_fidelity

    if sampler is None and folder is None:
        raise ValueError(f"model and folder cannot both be none")

    if folder is None:
        now = datetime.datetime.now().strftime("%d:%m:%Y-%H:%M:%S")
        folder = os.path.join(os.path.expanduser("~"), "evaluate_prettiness", f"{now}")
        os.makedirs(folder, exist_ok=True)

        idx = 0
        for _ in tqdm.tqdm(range(n // batch_size), desc="spawn samples"):
            batch = sampler(batch_size=batch_size).detach().cpu().numpy()
            if batch.shape[1] == 3:
                batch = batch.transpose((0, 2, 3, 1))
            for img in batch:
                img_path = os.path.join(folder, f"{idx:06d}.png")
                plt.imsave(img_path, img)
                idx += 1

    stats = torch_fidelity.calculate_metrics(folder, input_2, isc=isc, fid=fid, kid=kid)
    if clean_afterwards:
        shutil.rmtree(folder)
    return stats


def coos2adj(coos: Sequence[torch.Tensor], lengths: torch.Tensor, device=None):
    """Convert coordinate format tensor/list into an adjacency matrix.

    Args:
        coos: A sequence of index tensors each of size (T, 2).
        lengths: A tensor for the size of each entry of size (batch_size,).

    Returns:
        A single tensor of size (batch_size, max(lengths), max(lengths)) with 1s
        at position with arcs and 0s otherwise.
    """
    for possibly_tensor in (coos[0], lengths):
        if torch.is_tensor(possibly_tensor) and device is None:
            device = possibly_tensor.device
            break

    N = max(lengths)
    if isinstance(N, torch.Tensor):
        N = N.item()

    return torch.stack(
        [
            torch.sparse_coo_tensor(
                size=(N, N),
                indices=coo.t().to(device),
                values=torch.ones(size=(len(coo),), dtype=torch.long, device=device),
            ).to_dense()
            for coo in coos
        ],
        dim=0,
    )


def select_activation(activation="softplus"):
    # Avoid materializing the objects; just return the constructors.
    return {
        "softplus": nn.Softplus,
        "swish": Swish,
        "mish": Mish,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }[activation]


def select_optimizer(optimizer):
    def trim_dict(dictionary, keys):
        """Only grab specific keys from dictionary.

        Useful to allow arbitrary kwargs be accepted so that constructor doesn't break.
        """
        return {k: dictionary[k] for k in keys if k in dictionary}

    def optimizer_factory(params, **kwargs):
        if optimizer == "adam":
            keys = ("lr", "betas", "eps", "weight_decay", "amsgrad")
            return optim.Adam(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "sgd":
            keys = ("lr", "momentum", "dampening", "weight_decay", "nesterov")
            return optim.SGD(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adagrad":
            keys = (
                "lr",
                "lr_decay",
                "weight_decay",
                "initial_accumulator_value",
                "eps",
            )
            return optim.Adagrad(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adamax":
            keys = ("lr", "betas", "eps", "weight_decay")
            return optim.Adamax(params=params, **trim_dict(kwargs, keys))
        elif optimizer == "adadelta":
            keys = ("lr", "rho", "eps", "weight_decay")
            return optim.Adadelta(params=params, **trim_dict(kwargs, keys))
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    return optimizer_factory


# Helper functions that mimic tensorflow variants.
# Properly tested against tensorflow==2.3.1
def scatter_nd(indices, updates, shape=None, out=None, accumulate=True, inplace=True):
    if shape is None and out is None:
        raise ValueError("`out` and `shape` cannot both be `None` for `scatter_nd`.")

    if out is None:
        out = torch.zeros(size=shape, device=updates.device, dtype=updates.dtype)

    # `index_put` fails with non-contiguous tensors and produces uninterpretable error messages.
    if not out.is_contiguous():
        out = out.contiguous()

    if inplace:
        out.index_put_(
            indices=[indices[:, i] for i in range(indices.size(1))],
            values=updates,  # noqa
            accumulate=accumulate,
        )  # noqa
    else:
        out = out.index_put(
            indices=[indices[:, i] for i in range(indices.size(1))],
            values=updates,  # noqa
            accumulate=accumulate,
        )  # noqa
    return out


def cosine_similarity(t1, t2):
    return (t1 * t2).sum() / (t1.norm() * t2.norm())


def unravel_index(indices, shape):
    """Mimics np.unravel_index.

    See Also
        https://github.com/pytorch/pytorch/issues/35674
    """
    unraveled_coords = []
    for dim in reversed(shape):
        unraveled_coords.append(indices % dim)
        indices = indices // dim
    return torch.stack(unraveled_coords[::-1], dim=-1)


def topk(input, k, dim=-1, largest=True, sorted=True):
    """Returns multi-dim indices.

    See Also
        https://stackoverflow.com/questions/64241325/top-k-indices-of-a-multi-dimensional-tensor
    """
    v, i = torch.topk(input.flatten(), k=k, dim=dim, largest=largest, sorted=sorted)
    return v, unravel_index(i, input.size())


def retrieval_scores(
    tp: Union[torch.Tensor, np.ndarray],
    fp: Union[torch.Tensor, np.ndarray],
    fn: Union[torch.Tensor, np.ndarray],
):
    """Compute precision, recall, F1."""

    def _stable_div(x, y):
        """Returns 0 if x == 0, else x / y."""
        if not isinstance(x, np.ndarray):
            return 0.0 if x == 0.0 else (x / y)
        return np.where(x == 0, 0.0, x / y)

    tp, fp, fn = tuple(to_numpy(t) for t in (tp, fp, fn))
    precision = _stable_div(tp, tp + fp)
    recall = _stable_div(tp, tp + fn)
    f1 = _stable_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


# EMA model averaging.
def AverageModel(model: nn.Module, avg_fn: Union[str, Callable] = "ema", **kwargs):
    """Thin wrapper around `torch.optim.swa_utils.AveragedModel`."""
    if not callable(avg_fn):
        if avg_fn == "ema":
            gamma = kwargs.pop("gamma", 0.999)

            def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return (
                    gamma * averaged_model_parameter + (1.0 - gamma) * model_parameter
                )

            avg_fn = ema_avg_fn
        elif avg_fn == "warmup_ema":
            # From
            #   Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks."
            #   International conference on machine learning. PMLR, 2019.
            decay_rate = kwargs.pop("decay_rate", 0.9999)

            def warmup_ema_avg_fn(
                averaged_model_parameter, model_parameter, num_averaged
            ):
                gamma = min(decay_rate, (1 + num_averaged) / (10 + num_averaged))
                return (
                    gamma * averaged_model_parameter + (1.0 - gamma) * model_parameter
                )

            avg_fn = warmup_ema_avg_fn
        else:
            raise ValueError(f"Unknown average function: {avg_fn}.")
    return torch.optim.swa_utils.AveragedModel(model, avg_fn=avg_fn, **kwargs)


def denormalize(
    x: torch.Tensor, mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Unnormalize image for `torchvision.utils.save_image`."""
    # (bsz, n_channels, nh, hw) -> (n_channels, nh, nw, bsz).
    is_single_example = x.dim() == 3
    if is_single_example:
        x = x[None, ...]

    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # (n_channels, nh, nw, bsz) -> (bsz, n_channels, nh, hw).
    out = torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

    if is_single_example:
        return out[0]
    else:
        return out


# Plotting.
def plot_wrapper(*args, suffixes: Optional[Sequence] = None, **kwargs):
    """Allows specifying paths with multiple suffixes."""
    img_path = kwargs.pop("img_path", None)
    if img_path is None:
        return plot(*args, **kwargs)  # Directly plot.
    else:
        if suffixes is None:
            return plot(
                *args, img_path=img_path, **kwargs
            )  # Plot with img_path directly.
        else:
            # Append suffix to img_path.
            for suffix in suffixes:
                this_img_path = img_path + suffix
                plot(*args, img_path=this_img_path, **kwargs)


def plot(
    img_path: Optional[str] = None,
    plots: Sequence = (),
    steps: Sequence = (),
    vlines: Sequence = (),
    hlines: Sequence = (),
    scatters: Sequence = (),
    hists: Sequence = (),
    errorbars: Sequence = (),
    bars: Sequence = (),
    fill_betweens: Sequence = (),
    annotates: Sequence = (),
    stems: Sequence = (),
    options: Optional[Dict] = None,
    plots2: Sequence = (),
    steps2: Sequence = (),
    vlines2: Sequence = (),
    hlines2: Sequence = (),
    scatters2: Sequence = (),
    hists2: Sequence = (),
    errorbars2: Sequence = (),
    bars2: Sequence = (),
    fill_betweens2: Sequence = (),
    annotates2=(),
    stems2: Sequence = (),
    options2: Optional[Dict] = None,
    legend_options: Optional[Dict] = None,
    disable_legend: bool = False,
    finalized: bool = True,
    dpi: Optional[int] = None,
):
    """A multi-functional plotter to reduce boilerplate.

    Good features of this plotter are:
        1): Tweaked dpi.
        2): Enabled tight_layout.
        3): Plot closing.
        4): Twin plots.

    Args:
        img_path (str): A path to the place where the image should be written.
        plots (list of dict, optional): A list of curves that needs `plt.plot`.
        steps (list of dict, optional): A list of curves that needs `plt.step`.
        vlines (list of dict, optional): A list of vertical lines that needs `plt.vline`.
        scatters (list of dict, optional): A list of scatter plots that needs `plt.scatter`.
        hists (list of histograms, optional): A list of histograms that needs `plt.hist`.
        errorbars (list of errorbars, optional): A list of errorbars that needs `plt.errorbar`.
        bars (list of dict, optional): A list of bars that needs `plt.bar`.
        fill_betweens: (list of dict, optional): A list of shaded regions; kwargs: 'x', 'y1', 'y2'.
        options (dict, optional): A dictionary of optional arguments, such as title, xlabel, ylabel, etc.

        plots2: Same format as above, but for twin plot.
        steps2: Same format as above, but for twin plot.
        vlines2: Same format as above, but for twin plot.
        scatters2: Same format as above, but for twin plot.
        hists2: Same format as above, but for twin plot.
        errorbars2: Same format as above, but for twin plot.
        bars2: Same format as above, but for twin plot.
        fill_betweens2: Same format as above, but for twin plot.
        options2: Same format as above, but for twin plot.

        legend_options (dict, optional): A dictionary for kwargs passed to `ax.legend` or `plt.legend`.
            For the list of args/kwargs, see https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        disable_legend (bool): Remove the legend.
        finalized (bool, optional): Show or save the figure if True; otherwise the figure is still modifiable.

    Returns:
        Nothing.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="darkgrid")
    except ModuleNotFoundError:
        logging.warning(f"Unable to find `seaborn`, reverting to solely matplotlib.")

    if dpi is None:
        if img_path is None:
            dpi = 100
        else:
            dpi = 300

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111)
    if any(len(i) > 0 for i in (plots2, scatters2, hists2, errorbars2, bars2)):
        ax2 = ax.twinx()
    else:
        ax2 = None

    _plot(
        ax=ax,
        plots=plots,
        steps=steps,
        vlines=vlines,
        hlines=hlines,
        errorbars=errorbars,
        scatters=scatters,
        hists=hists,
        bars=bars,
        fill_betweens=fill_betweens,
        options=options,
        annotates=annotates,
        stems=stems,
    )

    # Twin-x plot: Share xaxis.
    if ax2 is not None:
        _plot(
            ax=ax2,
            plots=plots2,
            steps=steps2,
            vlines=vlines2,
            hlines=hlines2,
            scatters=scatters2,
            hists=hists2,
            errorbars=errorbars2,
            bars=bars2,
            fill_betweens=fill_betweens2,
            options=options2,
            annotates=annotates2,
            stems=stems2,
        )

    if legend_options is None:
        legend_options = dict(fontsize=13, framealpha=0.6)
    legend = ax.legend(**legend_options)
    if ax2 is not None:
        # Remove first legend then draw again to prevent second plot covering it.
        # https://stackoverflow.com/questions/29010078/matplotlib-data-being-plotted-over-legend-when-using-twinx
        legend.remove()
        if not disable_legend:
            ax2.legend(**legend_options)
            ax2.add_artist(legend)
    else:
        if disable_legend:
            legend.remove()

    plt.tight_layout()

    if finalized:
        if img_path is None:
            plt.show()
        else:
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            plt.savefig(img_path)
        plt.close()


def _feed_args(options, key, func):
    if key in options:
        params = options[key]
        if type(params) == dict:
            func(**params)
        elif type(params) in (list, tuple):
            func(*params)
        else:
            func(params)


def _plot(
    ax,
    plots,
    steps,
    vlines,
    hlines,
    errorbars,
    scatters,
    hists,
    bars,
    fill_betweens,
    options,
    annotates,
    stems,
):
    if options is None:
        options = dict()

    possible_options = {
        "xscale",
        "yscale",
        "xlabel",
        "ylabel",
        "xlabel_color",
        "ylabel_color",
        "title",
        "xlim",
        "ylim",
        "xticks",
        "yticks",
        "xticklabels",
        "yticklabels",
        "tick_params",
    }
    for key in options:
        if key not in possible_options:
            logging.warning(f"Unknown option fed to `_plot`: {key}")

    # Fix default font sizes for xylabels.
    if "xlabel" in options and not isinstance(options["xlabel"], dict):
        options["xlabel"] = dict(xlabel=options["xlabel"], fontdict=dict(size=18))
    if "ylabel" in options and not isinstance(options["ylabel"], dict):
        options["ylabel"] = dict(ylabel=options["ylabel"], fontdict=dict(size=18))

    _feed_args(options, "xscale", ax.set_xscale)
    _feed_args(options, "yscale", ax.set_yscale)
    _feed_args(options, "xlabel", ax.set_xlabel)
    _feed_args(options, "ylabel", ax.set_ylabel)
    _feed_args(options, "xlabel_color", ax.xaxis.label.set_color)
    _feed_args(options, "ylabel_color", ax.yaxis.label.set_color)
    _feed_args(options, "title", ax.set_title)
    _feed_args(options, "xlim", ax.set_xlim)
    _feed_args(options, "ylim", ax.set_ylim)
    _feed_args(options, "xticks", ax.set_xticks)
    _feed_args(options, "yticks", ax.set_yticks)
    _feed_args(options, "xticklabels", ax.set_xticklabels)
    _feed_args(options, "yticklabels", ax.set_yticklabels)
    _feed_args(options, "tick_params", ax.tick_params)

    for entry in plots:
        kwargs = {key: entry[key] for key in entry if key != "x" and key != "y"}
        kwargs.pop("aux", None)
        ax.plot(entry["x"], entry["y"], **kwargs)

    for entry in steps:
        kwargs = {key: entry[key] for key in entry if key != "x" and key != "y"}
        kwargs.pop("aux", None)
        ax.step(entry["x"], entry["y"], **kwargs)

    for entry in vlines:
        kwargs = {key: entry[key] for key in entry if key not in ("x", "ymin", "ymax")}
        kwargs.pop("aux", None)
        ax.vlines(entry["x"], entry["ymin"], entry["ymax"], **kwargs)

    for entry in hlines:
        kwargs = {key: entry[key] for key in entry if key not in ("y", "xmin", "xmax")}
        kwargs.pop("aux", None)
        ax.hlines(entry["y"], entry["xmin"], entry["xmax"], **kwargs)

    for entry in errorbars:
        kwargs = {key: entry[key] for key in entry if key != "x" and key != "y"}
        kwargs.pop("aux", None)
        if "capsize" not in kwargs:
            kwargs["capsize"] = 5
        ax.errorbar(entry["x"], entry["y"], **kwargs)

    for entry in scatters:
        kwargs = {key: entry[key] for key in entry if key != "x" and key != "y"}
        kwargs.pop("aux", None)
        ax.scatter(entry["x"], entry["y"], **kwargs)

    for entry in hists:
        kwargs = {key: entry[key] for key in entry if key != "x"}
        kwargs.pop("aux", None)
        ax.hist(entry["x"], **kwargs)

    for entry in fill_betweens:
        kwargs = {key: entry[key] for key in entry if key not in ("x", "y1", "y2")}
        kwargs.pop("aux", None)
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.4
        ax.fill_between(entry["x"], entry["y1"], entry["y2"], **kwargs)

    width = options.get("width", 0.2)
    for i, entry in enumerate(bars):
        kwargs = {key: entry[key] for key in entry if key != "x" and key != "height"}
        kwargs.pop("aux", None)
        bar_label = kwargs.pop(
            "bar_label", False
        )  # If True, give bars numbers at top (or right, if horizontal).
        # Each bar has width, the starting position is - (l-1) / 2 * width.
        x, height = [np.array(entry[key]) for key in ("x", "height")]
        bars = ax.bar(
            x - width * (len(bars) - 1) / 2 + width * i, height, width=width, **kwargs
        )
        if bar_label:
            ax.bar_label(bars)

    for entry in annotates:
        kwargs = copy.deepcopy(entry)
        text, xy = kwargs.pop("text"), kwargs.pop("xy")
        ax.annotate(text, xy, **kwargs)

    for entry in stems:
        kwargs = {key: entry[key] for key in entry if key not in ("locs", "heads")}
        ax.stem(entry["locs"], entry["heads"], **kwargs)

    return ax


def get_sns_colors(color=None, palette=None):
    import seaborn as sns

    palette = sns.color_palette(palette=palette)
    if color is None:
        return palette
    else:
        if color == "blue":
            return palette[0]
        elif color == "orange":
            return palette[1]
        elif color == "green":
            return palette[2]
        elif color == "red":
            return palette[3]
        elif color == "purple":
            return palette[4]
        elif color == "brown":
            return palette[5]
        elif color == "pink":
            return palette[6]
        elif color == "grey":
            return palette[7]
        elif color == "yellow":
            return palette[8]
        elif color == "cyan":
            return palette[9]
        else:
            raise ValueError(f"Unknown color: {color}")


def plot_side_by_side(
    figs1,
    figs2,
    nrows=8,
    ncols=1,
    img_path=None,
    dpi=300,
    title=None,
    left_title=None,
    right_title=None,
    frameon=True,
    max_batch_size=64,
):
    """Plot a dictionary of figures.

    Parameters
    ----------
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    figs1, figs2 = figs1.squeeze(), figs2.squeeze()
    if isinstance(figs1, torch.Tensor):
        figs1 = to_numpy(figs1)

    if isinstance(figs2, torch.Tensor):
        figs2 = to_numpy(figs2)

    assert figs1.shape == figs2.shape
    figs1, figs2 = figs1[:max_batch_size, ...], figs2[:max_batch_size, ...]

    if nrows * ncols < len(figs1):
        ncols = (len(figs1) + nrows - 1) // nrows
    assert nrows * ncols >= len(figs1)

    fig = plt.figure(dpi=dpi, frameon=frameon)
    outer = gridspec.GridSpec(1, 2, wspace=0.05, hspace=0.05)

    if left_title is not None:
        ax = plt.Subplot(fig, outer[0])
        ax.set_title(left_title)
        ax.axis("off")
        fig.add_subplot(ax)

    left_block = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=outer[0], wspace=0.0, hspace=0.0
    )
    for ind, item in enumerate(figs1):
        ax = plt.Subplot(fig, left_block[ind])
        ax.set_axis_off()
        ax.set_aspect("auto")

        if isinstance(figs1, dict):
            # `item` is the key.
            img = figs1[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    if right_title is not None:
        ax = plt.Subplot(fig, outer[1])
        ax.set_title(right_title)
        ax.axis("off")
        fig.add_subplot(ax)

    right_block = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=outer[1], wspace=0.0, hspace=0.0
    )
    for ind, item in enumerate(figs2):
        ax = plt.Subplot(fig, right_block[ind])
        ax.set_axis_off()
        ax.set_aspect("auto")

        if isinstance(figs2, dict):
            # `item` is the key.
            img = figs2[item]
            cmap = plt.gray() if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            ax.set_title(item)
        else:
            # `item` is the image.
            cmap = plt.gray() if len(item.shape) == 2 else None
            item = item.transpose(1, 2, 0) if item.shape[0] in (1, 3) else item
            ax.imshow(item, cmap=cmap)
        fig.add_subplot(ax)

    fig.suptitle(title)
    plt.savefig(img_path, bbox_inches="tight")
    plt.clf()
    plt.close()


# Shameless copy from https://matplotlib.org/3.5.0/gallery/images_contours_and_fields/image_annotated_heatmap.html
# TODO: These utils are inconvenient; need to modify in the future.
#  Also, kwargs should not have dict as default value. Terrible style!
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    import matplotlib.pyplot as plt

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            text = im.axes.text(j, i, data[i][j], **kw)
            texts.append(text)

    return texts


# Shameless copy from plottools https://pythonhosted.org/plottools/generated/plottools.zoom_axes.html
def zoom_axes(
    fig,
    ax,
    zoom_x,
    zoom_y,
    axes_x,
    axes_y,
    box=True,
    box_color="k",
    box_alpha=0.8,
    connect=True,
    connect_color="k",
    connect_alpha=0.3,
    spacing=4,
    tick_width=20,
    tick_height=12,
):
    """
    Creates a new axes which zooms in on a part of a given axes.

    A box is drawn around the area to be zoomed specified in data coordinates. A
    new empty axes is created at the specified location, supplied in data
    coordinates. The new axis limits are set so that they match the zoom box.

    The zoom box and axis can be connected with two lines, connecting the outer
    most corner points while leaving space for the axis ticks.


    Parameters
    ----------
    fig : matplotlib figure
        the figure in which to create a zoom axis

    ax : matplotlib axes
        the axis in which to create a zoom axis

    zoom_x : list
        [min, max] specifying the zooming horizontal area in data
        coordinates

    zoom_y : list
        [min, max] specifying the zooming vertical area in data coordinates

    axes_x : list
        [min, max] specifying the new axes horizontal location in data
        coordinates

    axes_y : list
        [min, max] specifying the new axes vertical location in data
        coordinates

    box : bool, optional
        specifies whether a box is drawn

    box_color : color string or tuple,optional
        specifies the box color

    box_alpha : number
        between 0 and 1, specifies the box alpha

    connect : bool, optional
        specifies whether the connecting lines are drawn

    connect_color : color string or tuple,optional
        specifies the connecting lines color

    connect_alpha : number
        between 0 and 1, specifies the connecting lines alpha

    spacing : number
        specifies the spacing between the box, axis and the connecting lines
        in points

    tick_width : number
        specifies the width of the tick labels in points to avoid drawing
        connecting lines through the tick labels

    tick_height : number
        specifies the height of the tick labels in points to avoid drawing
        connecting lines through the tick labels


    Returns
    -------
    ax_zoom : matplotlib axes
        the new axes

    Notes
    -----
    * Axes limits should not be changed after a zoom axes has been added
    * :code:`zoom_axes` does not give the expected results when used on a
      subfigure

    Examples
    --------
    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import plottools
        >>>
        >>> fig,ax = plt.subplots()
        >>> x = np.linspace(0,1,100)
        >>> y = 1-x + 0.02*(2*np.random.random(len(x))-1)
        >>> ax.plot(x,y)
        >>> ax_zoom = plottools.zoom_axes(fig,ax,[0.1,0.2],[0.8,0.9],[0.6,0.9],[0.6,0.9])
        >>> ax_zoom.plot(x,y)
        >>> plt.show()

    """

    import matplotlib.pyplot as plt

    plt.tight_layout()
    ax1_p0 = (ax.transData + fig.transFigure.inverted()).transform_point(
        (axes_x[0], axes_y[0])
    )
    ax1_p1 = (ax.transData + fig.transFigure.inverted()).transform_point(
        (axes_x[1], axes_y[1])
    )

    ax1 = plt.axes([ax1_p0[0], ax1_p0[1], ax1_p1[0] - ax1_p0[0], ax1_p1[1] - ax1_p0[1]])

    ax1.set_xlim(zoom_x)
    ax1.set_ylim(zoom_y)

    plt.xticks(fontsize=4)
    plt.yticks(fontsize=4)
    ax1.tick_params(axis="x", pad=3)
    ax1.tick_params(axis="y", pad=2)

    if box:
        ax.plot(
            [zoom_x[0], zoom_x[1], zoom_x[1], zoom_x[0], zoom_x[0]],
            [zoom_y[0], zoom_y[0], zoom_y[1], zoom_y[1], zoom_y[0]],
            color=box_color,
            alpha=box_alpha,
            linewidth=0.4,
        )

    if connect:
        # define a box of points of the axes and the zoom
        zoom_xx = [zoom_x[0], zoom_x[0], zoom_x[1], zoom_x[1]]
        zoom_yy = [zoom_y[0], zoom_y[1], zoom_y[1], zoom_y[0]]
        axes_xx = [axes_x[0], axes_x[0], axes_x[1], axes_x[1]]
        axes_yy = [axes_y[0], axes_y[1], axes_y[1], axes_y[0]]

        # determine which points to connect
        if axes_x[1] < zoom_x[1]:
            # left
            if axes_y[0] > zoom_y[0]:
                # top
                p1 = 0
                p2 = 2
            elif axes_y[1] < zoom_y[1]:
                # bottom
                p1 = 1
                p2 = 3
            else:
                # center
                p1 = 2
                p2 = 3

        elif axes_x[0] > zoom_x[0]:
            # right
            if axes_y[0] > zoom_y[0]:
                # top
                p1 = 1
                p2 = 3
            elif axes_y[1] < zoom_y[1]:
                # bottom
                p1 = 0
                p2 = 2
            else:
                # center
                p1 = 0
                p2 = 1

        else:
            # center
            if axes_y[0] > zoom_y[0]:
                # top
                p1 = 0
                p2 = 3
            elif axes_y[1] < zoom_y[1]:
                # bottom
                p1 = 1
                p2 = 2
            else:
                # center, the axes is over the zoom
                p1 = 0
                p2 = 0

        line1 = ([zoom_xx[p1], axes_xx[p1]], [zoom_yy[p1], axes_yy[p1]])
        line2 = ([zoom_xx[p2], axes_xx[p2]], [zoom_yy[p2], axes_yy[p2]])

        # estimate the width and height of the ticks
        tick_width = (ax.transData.inverted()).transform_point((tick_width, 0))[0] - (
            ax.transData.inverted()
        ).transform_point((0, 0))[0]
        tick_height = (ax.transData.inverted()).transform_point((0, tick_height))[1] - (
            ax.transData.inverted()
        ).transform_point((0, 0))[1]
        spacing = (ax.transData.inverted()).transform_point((spacing, 0))[0] - (
            ax.transData.inverted()
        ).transform_point((0, 0))[0]

        # create fictional boxes around the axes where no lines should be
        box_axes_x = [axes_x[0] - tick_width, axes_x[1] + spacing]
        box_axes_y = [axes_y[0] - tick_height, axes_y[1] + spacing]

        box_zoom_x = [zoom_x[0] - spacing, zoom_x[1] + spacing]
        box_zoom_y = [zoom_y[0] - spacing, zoom_y[1] + spacing]

        # cut the lines inside the boxes
        t = np.linspace(0, 1, 100)

        line1_cut = line1
        line2_cut = line2
        for tt in t:
            x = line1[0][0] * (1 - tt) + line1[0][1] * tt
            y = line1[1][0] * (1 - tt) + line1[1][1] * tt
            if (
                x <= box_zoom_x[0]
                or x >= box_zoom_x[1]
                or y <= box_zoom_y[0]
                or y >= box_zoom_y[1]
            ):
                line1_cut[0][0] = x
                line1_cut[1][0] = y
                break

        for tt in t[::-1]:
            x = line1[0][0] * (1 - tt) + line1[0][1] * tt
            y = line1[1][0] * (1 - tt) + line1[1][1] * tt
            if (x <= box_axes_x[0] or x >= box_axes_x[1]) or (
                y <= box_axes_y[0] or y >= box_axes_y[1]
            ):
                line1_cut[0][1] = x
                line1_cut[1][1] = y
                break

        for tt in t:
            x = line2[0][0] * (1 - tt) + line2[0][1] * tt
            y = line2[1][0] * (1 - tt) + line2[1][1] * tt
            if (x <= box_zoom_x[0] or x >= box_zoom_x[1]) or (
                y <= box_zoom_y[0] or y >= box_zoom_y[1]
            ):
                line2_cut[0][0] = x
                line2_cut[1][0] = y
                break

        for tt in t[::-1]:
            x = line2[0][0] * (1 - tt) + line2[0][1] * tt
            y = line2[1][0] * (1 - tt) + line2[1][1] * tt
            if (x <= box_axes_x[0] or x >= box_axes_x[1]) or (
                y <= box_axes_y[0] or y >= box_axes_y[1]
            ):
                line2_cut[0][1] = x
                line2_cut[1][1] = y
                break

        # draw the connecting lines
        ax.plot(
            line1_cut[0],
            line1_cut[1],
            color=connect_color,
            alpha=connect_alpha,
            linewidth=0.4,
        )
        ax.plot(
            line2_cut[0],
            line2_cut[1],
            color=connect_color,
            alpha=connect_alpha,
            linewidth=0.4,
        )

    return ax1


def make_mp4(img_paths: Sequence[str], out_path: str, fps: int):
    """Make an mp4 video from a list of images with paths specified."""
    import cv2  # Don't import unless absolutely necessary!

    if not out_path.endswith(".mp4"):
        raise ValueError(f"`out_path` must specify path to .mp4 file type")
    frame = cv2.imread(img_paths[0])
    cv2.imshow("video", frame)
    height, width, channels = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    for img_path in img_paths:
        frame = cv2.imread(img_path)
        out.write(frame)
        cv2.imshow("video", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):  # Hit `q` to exit
            break

    out.release()
    cv2.destroyAllWindows()


# Gradient checking.
def gradcheck(
    func: Callable,
    inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    modules: Optional[Union[nn.Module, Sequence[nn.Module]]] = (),
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    grad_inputs=False,
    gradgrad_inputs=False,
    grad_params=False,
    gradgrad_params=False,
):
    """Check grad and grad of grad wrt inputs and parameters of Modules.
    When `func` is vector-valued, the checks compare autodiff vjp against
    finite-difference vjp, where v is a sampled standard normal vector.
    This function is aimed to be as self-contained as possible so that it could
    be copied/pasted across different projects.
    Args:
        func (callable): A Python function that takes in a sequence of tensors
            (inputs) and a sequence of nn.Module (modules), and outputs a tensor
            or a sequence of tensors.
        inputs (sequence of Tensors): The input tensors.
        modules (sequence of nn.Module): The modules whose parameter gradient
            needs to be tested.
        eps (float, optional): Magnitude of two-sided finite difference
            perturbation.
        atol (float, optional): Absolute tolerance.
        rtol (float, optional): Relative tolerance.
        grad_inputs (bool, optional): Check gradients wrt inputs if True.
        gradgrad_inputs (bool, optional): Check gradients of gradients wrt
            inputs if True.
        grad_params (bool, optional): Check gradients wrt differentiable
            parameters of modules if True.
        gradgrad_params (bool, optional): Check gradients of gradients wrt
            differentiable parameters of modules if True.

    Returns:
        None.
    """

    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    if isinstance(modules, nn.Module):
        modules = (modules,)

    # Don't modify original objects.
    modules = tuple(copy.deepcopy(m) for m in modules)
    inputs = tuple(i.clone().requires_grad_() for i in inputs)

    func = _make_scalar_valued_func(func, inputs, modules)
    func_only_inputs = lambda *args: func(args, modules)  # noqa

    # Grad wrt inputs.
    if grad_inputs:
        torch.autograd.gradcheck(
            func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol
        )

    # Grad of grad wrt inputs.
    if gradgrad_inputs:
        torch.autograd.gradgradcheck(
            func_only_inputs, inputs, eps=eps, atol=atol, rtol=rtol
        )

    # Grad wrt params.
    if grad_params:
        params = [p for m in modules for p in m.parameters() if p.requires_grad]
        loss = func(inputs, modules)
        framework_grad = flatten(
            convert_none_to_zeros(
                torch.autograd.grad(loss, params, create_graph=True), params
            )
        )

        numerical_grad = []
        for param in params:
            flat_param = param.reshape(-1)
            for i in range(len(flat_param)):
                flat_param[i] += eps  # In-place.
                plus_eps = func(inputs, modules).detach()
                flat_param[i] -= eps

                flat_param[i] -= eps
                minus_eps = func(inputs, modules).detach()
                flat_param[i] += eps

                numerical_grad.append((plus_eps - minus_eps) / (2 * eps))
                del plus_eps, minus_eps
        numerical_grad = torch.stack(numerical_grad)
        torch.testing.assert_allclose(
            numerical_grad, framework_grad, rtol=rtol, atol=atol
        )

    # Grad of grad wrt params.
    if gradgrad_params:

        def func_high_order(inputs_, modules_):
            params_ = [p for m in modules for p in m.parameters() if p.requires_grad]
            grads = torch.autograd.grad(
                func(inputs_, modules_), params_, create_graph=True, allow_unused=True
            )
            return tuple(grad for grad in grads if grad is not None)

        gradcheck(
            func_high_order,
            inputs,
            modules,
            rtol=rtol,
            atol=atol,
            eps=eps,
            grad_params=True,
        )


def _make_scalar_valued_func(func, inputs, modules):
    outputs = func(inputs, modules)
    output_size = (
        outputs.numel() if torch.is_tensor(outputs) else sum(o.numel() for o in outputs)
    )

    if output_size > 1:
        # Define this outside `func_scalar_valued` so that random tensors are generated only once.
        grad_outputs = tuple(torch.randn_like(o) for o in outputs)

        def func_scalar_valued(inputs_, modules_):
            outputs_ = func(inputs_, modules_)
            return sum(
                (output * grad_output).sum()
                for output, grad_output in zip(outputs_, grad_outputs)
            )

        return func_scalar_valued

    return func


# Meters.
class Meter(abc.ABC):
    def __init__(self, init_val: Optional[float] = None, store_history=False):
        self._val = init_val
        self._his = []
        self._store_history = store_history

    @abc.abstractmethod
    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = to_numpy(x)
        if self._store_history:
            self._his.append(x)
        return x

    def item(self) -> float:
        return self._val


class EMAMeter(Meter):
    """Standard exponential moving average."""

    def __init__(self, gamma: Optional[float] = 0.99):
        super(EMAMeter, self).__init__()
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = super(EMAMeter, self).step(x)
        if self._val is None:
            self._val = x
        else:
            self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val


class AvgMeter(Meter):
    """Exact online averaging."""

    def __init__(self):
        super(AvgMeter, self).__init__()
        self._count = 0

    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = super(AvgMeter, self).step(x)
        if self._val is None:
            self._val = x
        else:
            self._val = self._val * self._count / (self._count + 1) + x / (
                self._count + 1
            )
        self._count += 1
        return self._val


class SumMeter(Meter):
    def __init__(self):
        super(SumMeter, self).__init__()

    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = super(SumMeter, self).step(x)
        if self._val is None:
            self._val = x
        else:
            self._val = self._val + x
        return self._val


class MaxMeter(Meter):
    def __init__(self):
        super(MaxMeter, self).__init__()

    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = super(MaxMeter, self).step(x)
        if self._val is None:
            self._val = x
        elif x > self._val:
            self._val = x
        return self._val


class MinMeter(Meter):
    def __init__(self):
        super(MinMeter, self).__init__()

    def step(self, x: Union[torch.Tensor, np.ndarray, float]):
        x = super(MinMeter, self).step(x)
        if self._val is None:
            self._val = x
        elif x < self._val:
            self._val = x
        return self._val


# Custom learning rate schedules.
def get_warmup_exp_decay_scheduler(
    optimizer: optim.Optimizer,
    num_warmup_steps: int,
    lr_decay_rate: Optional[float] = 0.99997,
    last_epoch: Optional[int] = -1,
):
    """Exponential decay schedule with linear warmup."""

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return lr_decay_rate ** (current_step - num_warmup_steps)

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_warmup_inverse_sqrt_scheduler(
    optimizer: optim.Optimizer,
    d_model: int,
    num_warmup_steps: Optional[int] = 4000,
    last_epoch: Optional[int] = -1,
    factor: Optional[float] = 1.0,
):
    """Inverse square root with linear warmup exactly as in transformers paper.

    Args:
        optimizer: Optimizer of choice.
        d_model: Size of transformer encoding.
        num_warmup_steps: Number of steps for linear warmup.
        last_epoch: Typical argument for lambda schedules.
        factor (float): A scalar factor applied to the default schedule. Defaults to 1., which is the original.

    Returns:
        A LambdaLR with corresponding schedule.
    """
    # Since LambdaLR multiplies the return value of the lambda_lr function with the lr,
    # we set lr to be 1.
    for param_group in optimizer.param_groups:
        if "lr" in param_group:
            param_group["lr"] = 1

    num_warmup_steps = max(
        num_warmup_steps, 1
    )  # To prevent raising zero to a negative power.

    def _lr_lambda(current_step):
        current_step += 1  # To prevent raising zero to a negative power.
        return (
            d_model**-0.5
            * min(current_step**-0.5, current_step * num_warmup_steps**-1.5)
            * factor
        )

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_linear_lr_scheduler(
    optimizer: optim.Optimizer,
    start_lr: float,
    end_lr: float,
    num_steps: int,
    last_epoch: Optional[int] = -1,
):
    """Simple linear scheduler from start_lr to end_lr.

    Becomes constant when current_step is larger than num_steps.
    """

    def _lr_lambda(current_step):
        return start_lr + (end_lr - start_lr) * (
            min(current_step, num_steps) / num_steps
        )

    return optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch=last_epoch)


def get_lr(optimizer: optim.Optimizer):
    return optimizer.param_groups[0]["lr"]


# Google cloud storage.
def gs_upload_from_path(local_path, remote_path=None, remove_local=True, timeout=480):
    """Uploads a single file to a remote gs bucket.

    Catches the exception and returns `False` if upload failed.
    """
    from google.cloud import storage  # noqa

    success = True

    if remote_path is None:
        remote_path = local_path
    remote_dir = remote_path.replace("gs://", "")
    bucket_id = remote_dir.split("/")[0]
    bucket_path = remote_dir[len("{}/".format(bucket_id)) :]

    try:
        bucket = storage.Client().bucket(bucket_id)
        blob = bucket.blob(bucket_path)
        blob.upload_from_filename(local_path, timeout=timeout)
    except MemoryError as memory_error:
        raise memory_error  # You don't want to catch this!!!
    except Exception as e:
        logging.warning(f"Failed uploading {local_path} to {remote_path}")
        logging.warning(f"Caught exception:\n{e}")
        success = False

    if remove_local:
        os.remove(local_path)

    return success


def gs_upload_from_dir(
    local_directory, remote_directory=None, remove_local=True, timeout=480
):
    if remote_directory is None:
        remote_directory = local_directory

    for root, _, files in os.walk(local_directory):
        for file in files:
            local_path = os.path.join(root, file)
            remote_path = remote_directory + local_path[len(local_directory) :]
            remote_path = str.lstrip(remote_path, "/")
            gs_upload_from_path(
                local_path, remote_path, remove_local=remove_local, timeout=timeout
            )


def gs_download_from_path(local_path, remote_path=None, timeout=480):
    from google.cloud import storage  # noqa

    success = True

    if remote_path is None:
        remote_path = local_path
    remote_dir = remote_path.replace("gs://", "")
    bucket_id = remote_dir.split("/")[0]
    bucket_path = remote_dir[len("{}/".format(bucket_id)) :]
    local_dir = os.path.dirname(local_path)
    os.makedirs(local_dir, exist_ok=True)

    try:
        bucket = storage.Client().bucket(bucket_id)
        blob = bucket.blob(bucket_path)
        blob.download_to_filename(local_path, timeout=timeout)
    except MemoryError as memory_error:
        raise memory_error  # You don't want to catch this!!!
    except Exception as e:
        logging.warning(f"Failed downloading {remote_path} to {local_path}")
        logging.warning(f"Caught exception:\n{e}")
        success = False

    return success


def gs_download_from_dir(remote_dir):
    from google.cloud import storage  # noqa

    remote_dir = remote_dir.replace("gs://", "")
    bucket_id = remote_dir.split("/")[0]
    bucket_dir = remote_dir[len("{}/".format(bucket_id)) :]

    bucket = storage.Client().bucket(bucket_id)
    blobs = bucket.list_blobs(prefix=bucket_dir)
    for blob in blobs:
        if blob.name.endswith("/"):  # Skip folders.
            continue
        # blob.name: folder/subfolder/file.
        tokens = blob.name.split("/")
        # Extract `local_dir` and `local_path`.
        local_dir_tokens = [bucket_id] + tokens[:-1]
        local_dir = os.path.join(*local_dir_tokens)

        local_path_tokens = [bucket_id] + tokens
        local_path = os.path.join(*local_path_tokens)
        os.makedirs(local_dir, exist_ok=True)
        blob.download_to_filename(local_path)


def gs_file_exists(remote_path):
    from google.cloud import storage  # noqa

    remote_dir = remote_path.replace("gs://", "")
    bucket_id = remote_dir.split("/")[0]
    bucket_path = remote_dir[len("{}/".format(bucket_id)) :]

    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob(bucket_path)
    return blob.exists()


def gs_listdir(remote_dir, full_path: Optional[bool] = False):
    from google.cloud import storage  # noqa

    remote_dir = remote_dir.replace("gs://", "")
    bucket_id = remote_dir.split("/")[0]
    bucket_dir = remote_dir[len("{}/".format(bucket_id)) :]

    bucket = storage.Client().bucket(bucket_id)
    blobs = bucket.list_blobs(prefix=bucket_dir)

    if full_path:
        return [os.path.join(bucket_id, blob.name) for blob in blobs]
    return blobs


# Timer.
class Timer(object):
    def __init__(
        self,
        msg=None,
        stream: Optional[Union[str, io.IOBase]] = "stderr",
        logging=False,
        level=logging.WARN,
    ):
        super(Timer, self).__init__()
        self.msg = msg
        if isinstance(stream, str):
            stream = {"stderr": sys.stderr, "stdout": sys.stdout}[stream]
        else:
            if not isinstance(stream, io.IOBase):
                raise ValueError(
                    f"Expected stream of type `io.IOBase`, but found: {type(stream)}"
                )
        self.stream = stream  # Output stream.
        self.logging = logging
        self.level = level

    def __enter__(self):
        self.now = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_elapse = time.perf_counter() - self.now
        msg = f"Time elapse={time_elapse:.6f}"
        if self.msg is not None:
            msg = f"{self.msg}: " + msg

        if self.logging:
            logging.log(level=self.level, msg=msg)
        else:
            print(msg, file=self.stream)


# Disable gc (e.g. for faster pickling).
# https://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram
class DisableGC(object):
    def __init__(self):
        super(DisableGC, self).__init__()

    def __enter__(self):
        gc.disable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.enable()


# Checkpoint.
def all_ckpts(dir_, sort=True):
    # Returns all checkpoint paths in the form of a used-once generator.
    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith("global_step_"), file_names)
    file_names = filter(lambda f: f.endswith(".ckpt"), file_names)
    file_names = map(lambda f: os.path.join(dir_, f), file_names)
    if sort:
        return sort_ckpts(file_names)
    return file_names


def sort_ckpts(file_names: Union[map, filter, list]):
    # Takes in an iterable (not necessarily a list); returns a list.
    if not isinstance(file_names, list):
        if not isinstance(file_names, collections.Iterable):
            raise ValueError
        file_names = list(file_names)
    # Avoid in-place ops that have side-effects.
    file_names_copy = file_names.copy()
    file_names_copy.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    return file_names_copy


def latest_ckpt(dir_, prefix="global_step_", suffix=".ckpt", num_digits=6):
    # Returns the path towards the latest ckpt. Returns `None` if no ckpt is found.
    # Assumes names are of the format `./parent_dir/global_step_i.ckpt`, where i is the index.
    # The prefix "global_step_" and suffix ".ckpt" must *both* be present in the path.
    def extract_id(name):
        assert isinstance(name, str)
        assert name.startswith(prefix) and name.endswith(suffix)
        name = name[len(prefix) :]
        name = name[: -len(suffix)]
        return int(name)

    file_names = os.listdir(dir_)
    file_names = filter(lambda f: f.startswith(prefix), file_names)
    file_names = filter(lambda f: f.endswith(suffix), file_names)
    idx = map(extract_id, file_names)
    idx = list(idx)
    if len(idx) == 0:
        print(f"Did not find any checkpoints in: {dir_}")
        return None

    latest_path = os.path.join(dir_, f"{prefix}{max(idx):0{num_digits}d}{suffix}")
    return latest_path


def save_ckpt(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    additional_state_dicts: Optional[
        Dict
    ] = None,  # Other state_dicts you might want to include.
    cloud_storage=False,  # cloud_storage is the legacy argument.
    to_gcs=False,
):
    # model, optimizer, scheduler are special parameters.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dicts = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
    }
    if additional_state_dicts is not None:
        for key, value in additional_state_dicts:
            state_dicts[key] = value

    # Save and upload.
    torch.save(state_dicts, path)
    if cloud_storage or to_gcs:
        gs_upload_from_path(path)


def load_ckpt(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    additional_state_objects: Optional[Dict] = None,
    verbose=True,
):
    if verbose:
        logging.warning(f"Loading checkpoint from {path}")

    state_dicts = torch.load(path)

    model.load_state_dict(state_dicts["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state_dicts["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state_dicts["scheduler"])
    if additional_state_objects is not None:
        for key, value in additional_state_objects:
            value.load_state_dict(state_dicts["key"])


# Data.
def get_data_stats(data_name):
    if data_name == "cifar10":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name == "cifar100":
        input_size = (3, 32, 32)
        classes = 100
    elif data_name in ("mnist", "kmnist", "fmnist"):
        input_size = (1, 28, 28)
        classes = 10
    elif data_name == "svhn":
        input_size = (3, 32, 32)
        classes = 10
    elif data_name in ("imagenet32", "imagenet64", "celebahq", "celeba_5bit"):
        input_size = (3, 32, 32)
        classes = None
    else:
        raise ValueError(f"Unknown data: {data_name}")
    return {"input_size": input_size, "classes": classes}


def dequantize(x, nvals=256):
    """[0, 1] -> [0, nvals] -> add uniform noise -> [0, 1]"""
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x


def get_loader(
    data_name,
    root=None,
    train_batch_size=128,
    test_batch_size=1024,
    pin_memory=True,
    num_workers=8,
    train_transform=None,
    test_transform=None,
    train_target_transform=None,
    test_target_transform=None,
    drop_last=True,
    shuffle=True,
    data_aug=True,
    padding_mode="constant",
    task="density",
    **kwargs,
):
    import torchvision as tv

    if task not in ("density", "classification", "hybrid"):
        raise ValueError(
            f"Unknown task: {task}. Expected one of `density`, `classification`, `hybrid`."
        )
    logging.warning(f"Creating loaders for data: {data_name}, task: {task}")

    if root is None:
        root = os.path.join(os.path.expanduser("~"), "data")
        os.makedirs(root, exist_ok=True)

    if data_name in ("cifar10", "cifar100"):
        if data_name == "cifar10":
            mean, std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        else:
            mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean, std),
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose(
                        [tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)]
                    )
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ToTensor(),
                            dequantize,
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.ToTensor(),
                            dequantize,
                        ]
                    )
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)]
                )
            else:  # `density`.
                test_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), dequantize]
                )

        if data_name == "cifar10":
            train_data = tv.datasets.CIFAR10(
                root,
                transform=train_transform,
                target_transform=train_target_transform,
                train=True,
                download=True,
            )
            test_data = tv.datasets.CIFAR10(
                root,
                transform=test_transform,
                target_transform=test_target_transform,
                train=False,
                download=True,
            )
        else:
            train_data = tv.datasets.CIFAR100(
                root,
                transform=train_transform,
                target_transform=train_target_transform,
                train=True,
                download=True,
            )
            test_data = tv.datasets.CIFAR100(
                root,
                transform=test_transform,
                target_transform=test_target_transform,
                train=False,
                download=True,
            )

    elif data_name == "svhn":
        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.ToTensor(),
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose([tv.transforms.ToTensor()])
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.ToTensor(),
                            dequantize,
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose(
                        [tv.transforms.ToTensor(), dequantize]
                    )
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose(
                    [
                        tv.transforms.ToTensor(),
                    ]
                )
            else:  # `density`.
                test_transform = tv.transforms.Compose(
                    [
                        tv.transforms.ToTensor(),
                        dequantize,
                    ]
                )
        train_data = tv.datasets.SVHN(
            root,
            transform=train_transform,
            target_transform=train_target_transform,
            split="train",
            download=True,
        )
        test_data = tv.datasets.SVHN(
            root,
            transform=test_transform,
            target_transform=test_target_transform,
            split="test",
            download=True,
        )

    elif data_name in ("mnist", "kmnist", "fmnist"):
        if train_transform is None:
            if task in ("classification", "hybrid"):
                train_transform = tv.transforms.Compose(
                    [
                        tv.transforms.ToTensor(),
                    ]
                )
            else:  # `density`.
                train_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), dequantize]
                )
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose(
                    [
                        tv.transforms.ToTensor(),
                    ]
                )
            else:  # `density`.
                test_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), dequantize]
                )

        if data_name == "mnist":
            train_data = tv.datasets.MNIST(
                root,
                train=True,
                transform=train_transform,
                target_transform=train_target_transform,
                download=True,
            )
            test_data = tv.datasets.MNIST(
                root,
                train=False,
                transform=test_transform,
                target_transform=test_target_transform,
                download=True,
            )
        elif data_name == "kmnist":  # `kmnist`
            train_data = tv.datasets.KMNIST(
                root,
                train=True,
                transform=train_transform,
                target_transform=train_target_transform,
                download=True,
            )
            test_data = tv.datasets.KMNIST(
                root,
                train=False,
                transform=test_transform,
                target_transform=test_target_transform,
                download=True,
            )
        else:  # `fmnist`
            train_data = tv.datasets.FashionMNIST(
                root,
                train=True,
                transform=train_transform,
                target_transform=train_target_transform,
                download=True,
            )
            test_data = tv.datasets.FashionMNIST(
                root,
                train=False,
                transform=test_transform,
                target_transform=test_target_transform,
                download=True,
            )
    elif data_name in ("cinic10", "cinic"):
        # Statistics from https://github.com/BayesWatch/cinic-10#data-loading
        mean, std = (
            (0.47889522, 0.47227842, 0.43047404),
            (
                0.24205776,
                0.23828046,
                0.25874835,
            ),
        )
        if train_transform is None:
            if task in ("classification", "hybrid"):
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ToTensor(),
                            tv.transforms.Normalize(mean, std),
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose(
                        [tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)]
                    )
            else:  # `density`.
                if data_aug:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.RandomCrop(
                                32, padding=4, padding_mode=padding_mode
                            ),
                            tv.transforms.RandomHorizontalFlip(),
                            tv.transforms.ToTensor(),
                            dequantize,
                        ]
                    )
                else:
                    train_transform = tv.transforms.Compose(
                        [
                            tv.transforms.ToTensor(),
                            dequantize,
                        ]
                    )
        if test_transform is None:
            if task in ("classification", "hybrid"):
                test_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), tv.transforms.Normalize(mean, std)]
                )
            else:  # `density`.
                test_transform = tv.transforms.Compose(
                    [tv.transforms.ToTensor(), dequantize]
                )

        # A bunch of hard-coded stuff that doesn't work if no access to bucket.
        cinic_path = os.path.join(root, "cinic-10")
        if not os.path.exists(cinic_path):
            cinic_link = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"
            os.system(f"wget -P {root} {cinic_link} --no-check-certificate")
            os.system(f"tar -xf {root}/CINIC-10.tar.gz")

        # Exclude the CIFAR-10 part in CINIC-10, since it's a hybrid of the original CIFAR-10 and Imagenet!
        if kwargs.get("exclude_cifar", False):
            is_valid_file = lambda _path: "cifar10" not in _path
        else:
            is_valid_file = None
        train_data = tv.datasets.ImageFolder(
            os.path.join(cinic_path, "train"),
            transform=train_transform,
            target_transform=train_target_transform,
            is_valid_file=is_valid_file,
        )
        test_data = tv.datasets.ImageFolder(
            os.path.join(cinic_path, "test"),
            transform=test_transform,
            target_transform=test_target_transform,
            is_valid_file=is_valid_file,
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {data_name}.")

    train_loader = data.DataLoader(
        train_data,
        batch_size=train_batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = data.DataLoader(
        test_data,
        batch_size=test_batch_size,
        drop_last=False,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def count_examples(loader: data.DataLoader):
    """Count the number of examples in a dataloader."""
    count = 0
    for batch in loader:
        unpacked = batch
        while not torch.is_tensor(unpacked):
            unpacked = unpacked[0]
        count += unpacked.size(0)
    return count


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: data.DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


class Comparator(object):
    def __init__(self, highest=True):
        super(Comparator, self).__init__()
        self._highest = highest
        self._best = (-sys.maxsize) if highest else sys.maxsize
        self._aux = {}

    def step(self, x, **kwargs) -> bool:
        """Update the internal state if `x` is better than the best recorded so far.

        Keyword Args are used to record auxiliary information.

        Returns:
            True if the internal state is updated; False otherwise.
        """
        if self._highest and x <= self._best:
            return False
        if not self._highest and x >= self._best:
            return False

        self._best = x
        self._aux = kwargs
        return True

    @property
    def val(self):
        return self._best, self._aux


class EarlyStopper(object):
    """An object that helps with early stopping."""

    def __init__(self, patience, want_max=True):
        super(EarlyStopper, self).__init__()
        self.patience = patience
        self.want_max = want_max
        self.flat_cnt = 0

        best = sys.maxsize
        if want_max:
            best = -best
        self.best = best

    def step(self, val) -> bool:
        """Given the current metric, return if the loop should break."""
        if self.patience is None:
            return False

        if self.want_max:
            if val <= self.best:
                self.flat_cnt += 1
                if self.flat_cnt >= self.patience:
                    return True
            else:
                self.best = val
                self.flat_cnt = 0
        else:
            if val >= self.best:
                self.flat_cnt += 1
                if self.flat_cnt >= self.patience:
                    return True
            else:
                self.best = val
                self.flat_cnt = 0
        return False


# Misc log sanitization.
def early_stopping(global_steps: list, metrics: list, tolerance: int, ascending: bool):
    """Find the index s.t. quant is best.

    Searches sequentially and stops when the traversed global steps in `gs` passes `tolerance` level.

    Args:
        global_steps: A list of global steps.
        metrics: A list of validation metrics.
        tolerance: The max number of steps before searching is stopped.
        ascending: Finds max metric if True; else finds the min.

    Returns:
        An integer index for the best position.
    """
    assert all(
        i > j for i, j in zip_(global_steps[1:], global_steps[:-1])
    ), "`global_steps` is not increasing."

    counts = 0  # The number of impatient steps.
    best = metrics[0]
    global_step_prev = global_steps[0]

    for i, (global_step, metric) in enumerate(zip_(global_steps[1:], metrics[1:])):
        if ascending:
            if metric <= best:
                counts += global_step - global_step_prev
                if counts >= tolerance:
                    break
            else:
                best = metric
                counts = 0
        else:
            if metric >= best:
                counts += global_step - global_step_prev
                if counts >= tolerance:
                    break
            else:
                best = metric
                counts = 0

        global_step_prev = global_step

    return metrics.index(best)


# Convenience aliases.
write_argparse = write_config
load_argparse = load_config
count_tensor_or_tensors_size = count_tensor_list_size


# Safe math operations.
def exp_(x):
    try:
        ans = math.exp(x)
    except OverflowError:
        ans = float("inf")
    return ans


# Run on cloud.
def extract_argument(cmd: str, arg="--train_dir"):
    lo = cmd.find(arg)
    start = lo + len(arg)
    end = None  # Until the last.
    for index in range(start, len(cmd) - 1):
        if cmd[index : index + 2] == "--":
            end = index
            break
    return cmd[start:end].strip()


def gpu_scheduler(
    commands: Sequence[str],
    wait_time_in_secs: int = 180,
    log=True,
    maxMemory=0.01,
    maxLoad=0.01,
    excludeID=(),
    excludeUUID=(),
):
    """Schedule jobs on a VM with several GPUs.

    Args:
        commands: Sequence of strings. Each string is a command of the format:
            python -m <script> <args>
            Notes:
                1) This command shouldn't contain CUDA_VISIBLE_DEVICES, since it gets added in this function.
                2) It is the responsibility of each command to get the wait/no wait right!
        wait_time_in_secs: The number of seconds to wait before scheduling the next job.
            It's always good to wait for a bit, since a job might not immediately start running a GPU.
        log: Write all logs to `train_dir/log.out` if True. So assumes command has `--train_dir` argument.
    """
    print(f"Scheduling {len(commands)} jobs...")
    import subprocess

    import GPUtil

    procs = []
    for job_id, command in enumerate(commands):
        empty_gpus = []
        num_times_failed_to_make_progress = -1
        while len(empty_gpus) == 0:
            num_times_failed_to_make_progress += 1
            if (
                num_times_failed_to_make_progress > 0
                and num_times_failed_to_make_progress % 240 == 0
            ):
                print(
                    f"Failed to fetch a GPU for {num_times_failed_to_make_progress} seconds."
                )
            # Don't use `getFirstAvailable`; it is very bad since it throws RuntimeError when no GPU is found.
            empty_gpus = GPUtil.getAvailable(
                order="first",
                maxLoad=maxLoad,
                maxMemory=maxMemory,
                limit=1,
                excludeID=excludeID,
                excludeUUID=excludeUUID,
            )
            time.sleep(1)
        print(f"empty gpus: {empty_gpus}")
        gpu_id = empty_gpus[0]

        command = f"export CUDA_VISIBLE_DEVICES={gpu_id}; {command}"
        command = command.strip()  # Need this strip to remove new line char.
        if log and "--train_dir" in command:
            # Get argument for `train_dir`.
            train_dir = extract_argument(command)
            log_path = os.path.join(train_dir, "log.out")
            command += f" > {log_path} 2>&1 "
            command = f"mkdir -p {train_dir}; \n{command}"

        # This doesn't wait.
        proc = subprocess.Popen(
            [command], shell=True, stdin=None, stdout=None, stderr=None, close_fds=True
        )
        procs.append(proc)
        print("command: ")
        print(command)
        print(f"scheduled job: {job_id} on gpu: {gpu_id}")

        # Give the program some time to be located on the GPU, before scheduling the next.
        time.sleep(wait_time_in_secs)

    return procs


def is_wandb_available():
    try:
        import wandb

        return True
    except ImportError:
        return False


# NLP.
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding in a smart way.

    Notes:
        For new tokens, the embedding value is the average of all old embedding vectors.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def e2e_metrics(
    reference_path: str,
    generation_path: str,
    out_path: Optional[str] = None,
    e2e_metrics_dir: Optional[str] = None,
    skip_coco=False,
    skip_mteval=False,
    conda_env_name="e2e_metrics",  # All evaluation is run a separate env so default env is not polluted.
    use_standard_format_for_bleu_and_rouge_l=True,
    conda_sh_path="~/miniconda3/etc/profile.d/conda.sh",
) -> Optional[dict]:
    """Run e2e-metrics.

    If repo doesn't exist, git clone and run from inside that repo.
    Repo url:
        https://github.com/lxuechen/e2e-metrics

    Args:
        reference_path: Path to the reference file.
        generation_path: Path to the generation file.
        out_path: Path to store json dump of results.
        e2e_metrics_dir: Directory of the e2e-metrics repo.
            If not given, defaults to cloning the repo to ${HOME}/evaluation/e2e-metrics.
    """
    if e2e_metrics_dir is None:
        e2e_metrics_dir = join(home, "evaluation", "e2e-metrics")

    if not pathexists(e2e_metrics_dir):
        e2e_metrics_dir_dirname = os.path.dirname(e2e_metrics_dir)
        os.makedirs(e2e_metrics_dir_dirname, exist_ok=True)
        # Be sure to use my fork; I fixed various annoying issues.
        os.system(
            f"cd {e2e_metrics_dir_dirname}; git clone https://github.com/lxuechen/e2e-metrics;"
        )

    # Check if env exists. If not, then create new environment with given name.
    conda_env_exists = os.system(
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name}'"
    )
    if conda_env_exists != 0:
        create_conda_env_cmd = f"conda create -n {conda_env_name} python=3.7 -y"
        os.system(f"bash -c 'source {conda_sh_path} && {create_conda_env_cmd}'")

    # Install the requirements into conda env.
    cmd = f"cd {e2e_metrics_dir}; pip install -r requirements.txt"
    cmd = (
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name} && {cmd}'"
    )
    os.system(cmd)

    # Run evaluation from within the repo.
    cmd = (
        f"cd {e2e_metrics_dir}; "
        f"./measure_scores.py {reference_path} {generation_path} "
        f"    --skip_coco {skip_coco} "
        f"    --skip_mteval {skip_mteval} "
        f"    --python True "
    )
    if out_path is not None:
        makedirs(dirname(out_path), exist_ok=True)
        cmd += f"    --out_path {out_path} "
    cmd = (
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name} && {cmd}'"
    )

    os.system(cmd)

    if out_path is not None:
        numbers = jload(out_path)
        if use_standard_format_for_bleu_and_rouge_l:
            numbers["BLEU"] = numbers["BLEU"] * 100.0
            numbers["ROUGE_L"] = numbers["ROUGE_L"] * 100.0
            jdump(numbers, out_path)
        return jload(out_path)


def gem_metrics(
    reference_path: str,
    generation_path: str,
    out_path: Optional[str] = None,
    gem_metrics_dir: Optional[str] = None,
    metric_list=(
        "bleu",
        "rouge",
        "nist",
        "bertscore",
    ),
    heavy_metrics=False,
    conda_env_name="gem_metrics",  # All evaluation is run a separate env so default env is not polluted.
    conda_sh_path="~/miniconda3/etc/profile.d/conda.sh",
):
    if gem_metrics_dir is None:
        gem_metrics_dir = join(home, "evaluation", "GEM-metrics")

    if not pathexists(gem_metrics_dir):
        gem_metrics_dir_dirname = os.path.dirname(gem_metrics_dir)
        os.makedirs(gem_metrics_dir_dirname, exist_ok=True)
        os.system(
            f"cd {gem_metrics_dir_dirname}; "
            f"git clone https://github.com/GEM-benchmark/GEM-metrics; "
        )

    # Check if env exists. If not, then create new environment with given name.
    conda_env_exists = os.system(
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name}'"
    )
    if conda_env_exists != 0:
        cmd = f"conda create -n {conda_env_name} python=3.7 -y"
        os.system(f"bash -c 'source {conda_sh_path} && {cmd}'")

    # Install the requirements into conda env.
    cmd = (
        f"cd {gem_metrics_dir}; "
        f"pip install -r requirements.txt -r requirements-heavy.txt; "
        f"pip uninstall -y torch torchvision torchaudio; "
        # TODO: This is only a temp sol'n.
        #  The annoying issue of can't get torch+cuda installed correctly through requirements.txt...
        f"pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113; "
    )
    cmd = (
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name} && {cmd}'"
    )
    os.system(cmd)

    metric_list = " ".join(metric_list)
    cmd = (
        f"cd {gem_metrics_dir}; "
        f"./run_metrics.py {generation_path} -r {reference_path} --metric-list {metric_list} "
    )
    if out_path is not None:
        makedirs(dirname(out_path), exist_ok=True)
        cmd += f"-o {out_path} "
    if heavy_metrics:
        cmd += "--heavy-metrics "
    cmd = (
        f"bash -c 'source {conda_sh_path} && conda activate {conda_env_name} && {cmd}'"
    )

    os.system(cmd)

    if out_path is not None:
        return jload(out_path)


# NLP stuff.
def get_seq_len_dist_enc_dec(
    tokenizer,
    hf_datadict,
    enc_key: str,
    dec_key: str,
    splits=("train", "test"),
    verbose=True,
    maxsize=sys.maxsize,
):
    """Get the sequence length distribution of an encoder-decoder setting.

    hf_datadict takes form {split: [{enc_key: str, dec_key: str}, ...]}
    """
    _increment_key_for_dict = (
        lambda d, k: d.update({k: d[k] + 1}) if k in d else d.update({k: 1})
    )

    result = dict()
    for split in splits:
        enc_counts_map, dec_counts_map, cat_counts_map = tuple(
            dict() for _ in range(3)
        )  # seq_len -> count
        enc_counts_list, dec_counts_list, cat_counts_list = tuple([] for _ in range(3))
        for idx, instance in tqdm.tqdm(
            enumerate(hf_datadict[split]),
            desc=split,
            total=min(maxsize, len(hf_datadict[split])),
        ):
            if idx >= maxsize:
                break

            enc_len = len(tokenizer(instance[enc_key])["input_ids"])
            dec_len = len(tokenizer(instance[dec_key])["input_ids"])
            cat_len = enc_len + dec_len  # length of concatenated sequence

            _increment_key_for_dict(enc_counts_map, enc_len)
            _increment_key_for_dict(dec_counts_map, dec_len)
            _increment_key_for_dict(cat_counts_map, cat_len)

            enc_counts_list.append(enc_len)
            dec_counts_list.append(dec_len)
            cat_counts_list.append(cat_len)

        result[split] = dict(
            enc_counts_map=enc_counts_map,
            dec_counts_map=dec_counts_map,
            cat_counts_map=cat_counts_map,
            enc_counts_list=enc_counts_list,
            dec_counts_list=dec_counts_list,
            cat_counts_list=cat_counts_list,
        )
        if verbose:
            print(jdumps(result, indent=4))
            plot(
                hists=[
                    dict(x=enc_counts_list, label="enc"),
                    dict(x=dec_counts_list, label="dec"),
                    dict(x=cat_counts_list, label="cat"),
                ],
                options=dict(title=split),
            )
    return result


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.

    Shameless copy from
        https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    @torch.no_grad()
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
