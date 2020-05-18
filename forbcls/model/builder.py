from forbcls.utils import build_from_cfg

from .registry import BACKBONES, MODELS


def build_backbone(cfg, default_args=None):
    backbone = build_from_cfg(cfg, BACKBONES, default_args)
    return backbone

def build_model(cfg, default_args=None):
    model = build_from_cfg(cfg, MODELS, default_args)
    return model
