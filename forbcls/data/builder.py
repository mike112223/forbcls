import torch.utils.data as torch_data

from forbcls.utils import build_from_cfg
from .dataset.transform import Compose

from .registry import DATASETS, TRANSFORMS


def build_dataloader(cfg, default_args):
    loader = build_from_cfg(cfg, torch_data, default_args, 'module')
    return loader


def build_dataset(cfg, default_args=None):
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset


def build_transform(cfg):
    tfs = []
    for icfg in cfg:
        tf = build_from_cfg(icfg, TRANSFORMS)
        tfs.append(tf)
    aug = Compose(tfs)

    return aug
