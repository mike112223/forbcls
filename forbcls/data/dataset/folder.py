import os

import numpy as np

from .base import BaseDataset
from ..registry import DATASETS


@DATASETS.register_module
class Folder(BaseDataset):
    allowed_class_modes = {'categorical', None}
    def __init__(self, directory, transforms=None, class_mode='categorical'):
        self.directory = directory
        self.transforms = transforms
        self.class_mode = class_mode
        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))

        self.get_classes()
        self.samples = self.get_samples()

    def get_classes(self):
        classes = []
        for subdir in sorted(os.listdir(self.directory)):
            if os.path.isdir(os.path.join(self.directory, subdir)):
                classes.append(subdir)
        self.num_classes = len(classes)
        self.class2index = dict(zip(classes, range(len(classes))))

    def get_samples(self):
        samples = []
        for k in self.class2index:
            img_dir = os.path.join(self.directory, k)
            img_paths = os.listdir(img_dir)
            samples.extend([[os.path.join(img_dir, p), self.class2index[k]] for p in sorted(img_paths)])

        return samples

    def __len__(self):
        return len(self.samples)
