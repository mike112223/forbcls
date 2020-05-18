import os
import sys
sys.path.insert(0, os.path.abspath('../forbcls'))

from collections import defaultdict

import cv2
import numpy as np

from forbcls.utils import Config
from forbcls.data import build_dataset, build_transform, build_dataloader


def main():

    cfg_fp = os.path.join(os.path.abspath('config'), 'test.py')
    cfg = Config.fromfile(cfg_fp)

    val_tf = build_transform(cfg['data']['val']['transforms'])
    val_dataset = build_dataset(cfg['data']['val']['dataset'], dict(transforms=val_tf))

    val_loader = build_dataloader(cfg['data']['val']['loader'], dict(dataset=val_dataset))

    num_label = defaultdict(int)
    for i, batch in enumerate(val_loader):

        img, label = batch

        img = img.numpy()[0]
        label = label.numpy()[0]

        num_label[label] += 1

        save_path = os.path.join('tests/test_imgs/', '{}_{}.jpg'.format(i, label))
        print(img.shape)

        cv2.imwrite(save_path, img)

        if i == 30:
            break

    print(num_label)


if __name__ == '__main__':
    main()
