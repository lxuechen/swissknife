"""
Convert the tf checkpoints to torch.

You need Python<3.7 to enable tf==1.15.4!!! This is confusing.
"""
import os

import fire
import tqdm

from download import available_simclr_models, simclr_categories


def main(category="pretrained"):
    category_hash = simclr_categories[category]
    for model in tqdm.tqdm(available_simclr_models):
        tf_path = os.path.join('.', model, 'model.ckpt-250228')
        os.system(f'python ./convert.py {tf_path} --ema')


if __name__ == "__main__":
    fire.Fire(main)
