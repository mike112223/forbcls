import os
import sys
from collections import defaultdict
import random
sys.path.insert(0, os.path.abspath('../forbcls'))

import cv2
import numpy as np

from forbcls.data import build_dataset, build_transform, build_dataloader

from keras_preprocessing.image import ImageDataGenerator


np.random.seed(123)

def build_tensorflow_loader():

    val_dir = "data/frame_test"
    val_datagen = ImageDataGenerator(
        # rescale=1. / 255,
        # rotation_range=30,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # channel_shift_range=100,
        # fill_mode='nearest',
        # shear_range=0.2,
        # zoom_range=0.2,
        # horizontal_flip=True
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(331, 284),
        batch_size=1,
        shuffle=False,
        class_mode='categorical')

    return val_generator


def main():

    tf_loader = build_tensorflow_loader()

    num_label = defaultdict(int)

    for i, batch in enumerate(tf_loader):

        tf_img, tf_label = batch

        tf_img = tf_img[0][:, :, (2, 1, 0)]

        # import pdb
        # pdb.set_trace()

        print(np.sum(tf_img))

        cv2.imshow('tf', tf_img)

        cv2.waitKey(0)

        break

        # import pdb
        # pdb.set_trace()

if __name__ == '__main__':
    main()
