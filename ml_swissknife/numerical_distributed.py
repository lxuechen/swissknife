"""Numerical algorithms that could be run on multiple GPUs.

How does this work? Well, the obvious observation is that matmul can be run easily in parallel by splitting mats
into chunks.
"""
import logging
import math
from typing import Optional, Tuple, Union

import fire
import numpy as np
import torch
import tqdm
from ml_swissknife import utils
from torch.utils.data import DataLoader, TensorDataset, Dataset


def orthogonal_iteration(
    input_mat: Union[DataLoader, Dataset, torch.Tensor],
    k: int,
    num_power_iteration=1,
    disable_tqdm=False,
    dtype=torch.float,
    device: Optional[torch.device] = None,
    dump_dir=None,
    dim0_chunk_size=10,
    chunk_size=100,
    chunk_size_2=10,
    eval_steps=5,
):
    """Simultaneous iteration for finding eigenvectors with the largest eigenvalues in absolute value.

    The method is aka subspace iteration or orthogonal iteration.

    WARNING:
        - good reconstruction of the data does not imply converged eigenvalues!

    Args:
        input_mat: DataLoader or Dataset or torch.Tensor as data. Always assumes batch first.
        k: Number of principal components to return.
        num_power_iteration: Number of power iterations.
        disable_tqdm: If True, disable progress bar.
        device: torch.device; defaults to CPU if None.
        dump_dir: Directory to dump the sequence of results.
        dtype: Precision in string format.
        dim0_chunk_size: Size of chunks for dim0 -- the batch dimension of input_mat.
        chunk_size: Size of chunks for processing the dimension that loops over eigenvectors.
        chunk_size_2: Size of chunks for orthogonalization.
        eval_steps: Number of steps before a data reconstruction evaluation.

    Returns:
        eigenvectors: Tensor of selected basis of size (p, k).
        eigenvalues: Tensor of eigenvalues of data.T @ data of size (k,).
    """
    if isinstance(input_mat, torch.Tensor):
        input_mat = TensorDataset(input_mat)
    if isinstance(input_mat, Dataset):
        loader = DataLoader(
            dataset=input_mat,
            batch_size=dim0_chunk_size,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
    elif isinstance(input_mat, DataLoader):
        loader = input_mat
    else:
        raise ValueError(
            f"Expected `input_mat` to be of type `torch.utils.data.DataLoader` or `torch.Tensor`, "
            f"but found type={type(input_mat)}"
        )

    n = sum(batch.size(0) for batch, in loader)
    batch, = next(iter(loader))
    p = batch.size(1)
    k = min(k, p, n)
    eigenvectors = torch.randn(size=(p, k), dtype=dtype)  # This step will be very slow for large models.

    err_abs, err_rel = _check_error(
        loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
        device=device, disable_tqdm=disable_tqdm,
    )
    logging.warning(f"before iteration, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    for global_step in tqdm.tqdm(range(1, num_power_iteration + 1), desc="power iteration", disable=disable_tqdm):
        matrix = _mem_saving_matmul(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )
        eigenvectors = _orthogonalize(
            matrix=matrix, chunk_size_2=chunk_size_2,
            device=device, disable_tqdm=disable_tqdm
        )  # (p, k).
        eigenvalues = _eigenvectors_to_eigenvalues(
            loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
            device=device, disable_tqdm=disable_tqdm
        )

        if dump_dir is not None:
            utils.tsave(
                dict(eigenvalues=eigenvalues, eigenvectors=eigenvectors),
                utils.join(dump_dir, "all", f"global_step_{global_step:06d}.pt")
            )
            utils.tsave(
                dict(eigenvalues=eigenvalues),
                utils.join(dump_dir, "eigenvalues", f"global_step_{global_step:06d}.evals")
            )

        if global_step % eval_steps == 0:
            err_abs, err_rel = _check_error(
                loader=loader, eigenvectors=eigenvectors, chunk_size=chunk_size,
                device=device, disable_tqdm=disable_tqdm,
            )
            logging.warning(f"global_step: {global_step}, abs error: {err_abs:.6f}, rel error: {err_rel:.6f}")

    return eigenvalues, eigenvectors  # noqa


def _check_error(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    disable_tqdm: bool,
    **kwargs,
):
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
    devices = tuple(range(num_cuda_devices))

    evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
    evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))

    ref_abs = []
    err_abs = []
    for (batch,) in tqdm.tqdm(loader, desc="check error", disable=disable_tqdm):
        batch_recs = []
        for evec_chunk in evec_chunks:
            this_batch = batch.to(evec_chunk.device, non_blocking=True)
            batch_recs.append(
                torch.mm(evec_chunk, torch.mm(evec_chunk.T, this_batch.T)).T
            )

        batch_rec = batch_recs[0]
        for this_batch_rec in batch_recs[1:]:
            batch_rec += this_batch_rec.to(0)
        batch = batch.to(0)

        err_abs.append((batch - batch_rec).norm(2))
        ref_abs.append(batch.norm(2))

    ref_abs = torch.stack(ref_abs).norm(2)
    err_abs = torch.stack(err_abs).norm(2)
    err_rel = err_abs / ref_abs

    return err_abs.item(), err_rel.item()


def _mem_saving_matmul(
    loader: DataLoader,
    eigenvectors: torch.Tensor,
    disable_tqdm: bool,
    **kwargs,
):
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
    devices = tuple(range(num_cuda_devices))

    out = torch.zeros_like(eigenvectors)

    evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
    evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))

    chunk_num_cols = (0,) + tuple(evec_chunk.size(1) for evec_chunk in evec_chunks)
    chunk_num_cols_cumsum = np.cumsum(chunk_num_cols)
    chunk_col_ranges = tuple(utils.zip_(chunk_num_cols_cumsum[:-1], chunk_num_cols_cumsum[1:]))

    for (batch,) in tqdm.tqdm(loader, desc="batches", disable=disable_tqdm):
        outs = []
        for chunk in evec_chunks:
            batch = batch.to(chunk.device, non_blocking=True)
            outs.append(torch.mm(batch.T, torch.mm(batch, chunk)))

        outs = [o.cpu() for o in outs]
        for o, chunk_col_range in utils.zip_(outs, chunk_col_ranges):
            out[:, chunk_col_range[0]:chunk_col_range[1]] += o

    return out


def _orthogonalize(matrix, device, disable_tqdm: bool, chunk_size_2=20):
    if device.type == "cuda":
        devices = tuple(range(torch.cuda.device_count()))
    else:
        devices = (device,)

    matrix_chunks = torch.tensor_split(matrix, len(devices), dim=1)
    matrix_chunks = tuple(
        matrix_chunk.to(matrix_device) for matrix_chunk, matrix_device in utils.zip_(matrix_chunks, devices)
    )
    chunk_num_cols = (0,) + tuple(matrix_chunk.size(1) for matrix_chunk in matrix_chunks)
    chunk_num_cols_cumsum = np.cumsum(chunk_num_cols)
    chunk_col_ranges = tuple(utils.zip_(chunk_num_cols_cumsum[:-1], chunk_num_cols_cumsum[1:]))

    def col_idx_to_chunk_idx_and_offset(col_idx):
        """Returns the index of the matrix chunk and the offset to index into."""
        # k, offset = col_idx_to_chunk_idx_and_offset(col_idx)
        # col = matrix_chunks[k][offset]
        for k, (head, tail) in enumerate(chunk_col_ranges):
            if head <= col_idx < tail:
                offset = col_idx - head
                return k, offset
        assert False, "Internal error: Should not reach here!"

    def gram_schmidt_helper(col_, rest_):
        col_ = col_.to(rest_)
        start_idx = 0
        while start_idx < rest_.size(1):
            batch = rest_[:, start_idx:start_idx + chunk_size_2]
            batch -= torch.sum(col_ * batch, dim=0) * col_
            start_idx += chunk_size_2

    for i in tqdm.tqdm(range(matrix.size(1)), desc="orthogonalize", disable=disable_tqdm):
        k, offset = col_idx_to_chunk_idx_and_offset(i)
        matrix_chunk = matrix_chunks[k]

        col = matrix_chunk[:, offset:offset + 1]
        col /= col.norm(2)
        if i + 1 < matrix.size(1):
            # current matrix_chunk.
            rest = matrix_chunk[:, offset + 1:]
            gram_schmidt_helper(col, rest)

            # future matrix_chunk.
            for future_matrix_chunk in matrix_chunks[k + 1:]:
                rest = future_matrix_chunk
                gram_schmidt_helper(col, rest)
    return torch.cat(tuple(matrix_chunk.cpu() for matrix_chunk in matrix_chunks), dim=1)


def _eigenvectors_to_eigenvalues(
    loader: DataLoader, eigenvectors: torch.Tensor,
    disable_tqdm: bool,
    **kwargs,
):
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, "v2 is only supported in distributed settings."
    devices = tuple(range(num_cuda_devices))

    evec_chunks = torch.tensor_split(eigenvectors, len(devices), dim=1)
    evec_chunks = tuple(evec_chunk.to(device) for evec_chunk, device in utils.zip_(evec_chunks, devices))

    dens = [(chunk ** 2.).sum(dim=0) for chunk in evec_chunks]
    nums = [torch.zeros_like(den) for den in dens]
    for (batch,) in tqdm.tqdm(loader, disable=disable_tqdm, desc="evec2eval"):
        for chunk_id, chunk in enumerate(evec_chunks):
            # Don't override `batch` here to prevent weird bugs. Moving batches across GPUs introduces weird issues!
            gpu_batch = batch.to(chunk.device)
            vec = gpu_batch @ chunk  # (nj, ki).
            nums[chunk_id] += (vec ** 2.).sum(dim=0)
    nums = [num.cpu() for num in nums]
    dens = [den.cpu() for den in dens]
    return torch.cat(nums) / torch.cat(dens)
