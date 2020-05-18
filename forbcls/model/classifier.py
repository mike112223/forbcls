
import torch.nn as nn

from .registry import MODELS
from .builder import build_backbone


@MODELS.register_module
class Classifier(nn.Module):
    def __init__(self,
                 backbone,
                 pretrained=None,
                 progress=True):
        super(Classifier, self).__init__()

        self.backbone = build_backbone(backbone)

    def forward(self, img):
        return self.backbone(img)
