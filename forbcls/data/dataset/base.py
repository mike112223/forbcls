
import cv2
import numpy as np

from torch.utils.data import Dataset

from ..registry import DATASETS


@DATASETS.register_module
class BaseDataset(Dataset):

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        # pii = Image.open(path)
        # img = cv2.cvtColor(np.asarray(pii), cv2.COLOR_RGB2BGR)

        img = cv2.imread(path).astype(np.float64)

        if self.transforms:
            img, label = self.transforms(img, label)

        return img, label
