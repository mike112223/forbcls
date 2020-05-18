
import torch.nn as nn

from retina.utils import build_from_cfg
from .registry import LOSSES


def build_loss(cfg, default_args=None):
    loss = build_from_cfg(cfg, LOSSES, default_args)
    return loss

def build_torch_loss(cfg, default_args):
    loader = build_from_cfg(cfg, nn, default_args, 'module')
    return loader
