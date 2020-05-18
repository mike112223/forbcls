import os
import sys
from collections import defaultdict
import random
sys.path.insert(0, os.path.abspath('../forbcls'))

import cv2
import numpy as np

from forbcls.data import build_dataset, build_transform, build_dataloader


def build_pytorch_loader():
    img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
    data = dict(
        val=dict(
            dataset=dict(
                type='Folder',
                directory='data/test',
                class_mode='categorical',
            ),
            transforms=[
                dict(type='Resize', img_scale=(128, 128)),
                dict(
                    type='RandomAffineTransform',
                    rotate_range=30, rotate_ratio=1,
                    shear_range=.2, shear_ratio=1,
                    shift_range=0.2, shift_ratio=1,
                    zoom_range=0.2, zoom_ratio=1,
                ),
                dict(type='RandomChannelShift', shift_range=20, ratio=1.0),
                # dict(type='RandomFlip', flip_ratio=1.0),
                # dict(type='Normalize', **img_norm_cfg),
            ],
            loader=dict(
                type='DataLoader',
                batch_size=1,
                num_workers=1,
                shuffle=True,
                drop_last=True,
            ),
        )
    )

    val_tf = build_transform(data['val']['transforms'])
    val_dataset = build_dataset(data['val']['dataset'], dict(transforms=val_tf))

    val_loader = build_dataloader(data['val']['loader'], dict(dataset=val_dataset))

    return val_loader


def main():

    py_loader = build_pytorch_loader()

    num_label = defaultdict(int)

    for i, batch in enumerate(py_loader):

        py_img, py_label = batch

        py_img = py_img.numpy()[0].astype(np.uint8)

        # import pdb
        # pdb.set_trace()

        print(py_label)

        cv2.imshow('py', py_img)

        cv2.waitKey(0)

        # import pdb
        # pdb.set_trace()

if __name__ == '__main__':
    main()
