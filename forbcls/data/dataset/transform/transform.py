from collections.abc import Sequence

import cv2
import numpy as np
import torch

from forbcls.utils.misc import is_str

from ...registry import TRANSFORMS


CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}


CV2_BORDER_MODE = {
    'default': cv2.BORDER_DEFAULT,
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
            if img is None:
                return None, None
        return img, label


@TRANSFORMS.register_module
class ToTensor(object):
    @staticmethod
    def _to_tensor(data):
        """Convert objects of various python types to :obj:`torch.Tensor`.

        Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
        :class:`Sequence`, :class:`int` and :class:`float`.
        """
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not is_str(data):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError('type {} cannot be converted to tensor.'.format(
                type(data)))

    def __call__(self, img, label):
        img = self.to_tensor(img)
        label = self.to_tensor(label)
        return img, label


@TRANSFORMS.register_module
class Resize(object):
    def __init__(self, img_scale, keep_ratio=False, mode='nearest'):
        self.img_scale = img_scale
        if not isinstance(self.img_scale, tuple):
            raise TypeError('img scale must be tuple!')
        elif len(self.img_scale) != 2:
            raise ValueError('img scale must contains w and h!')

        self.keep_ratio = keep_ratio
        self.mode = CV2_MODE[mode]

    def _resize_img(self, img):
        h, w = img.shape[:2]
        if self.keep_ratio:
            scale_factor = min(self.img_scale[0] / w, self.img_scale[1] / h)
            new_size = (
                int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5)
            )
            resized_img = cv2.resize(img, new_size, interpolation=self.mode)
        else:
            resized_img = cv2.resize(
                img, self.img_scale, interpolation=self.mode
            )
            w_scale = self.img_scale[0] / w
            h_scale = self.img_scale[1] / h

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        return resized_img

    def __call__(self, img, label):
        return self._resize_img(img), label


@TRANSFORMS.register_module
class RandomFlip(object):
    def __init__(self, flip_ratio=0.5, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction

    def _flip_img(self, img):
        if self.direction == 'horizontal':
            flipped_img = np.flip(img, axis=1)
        else:
            flipped_img = np.flip(img, axis=0)
        return flipped_img

    def __call__(self, img, label):
        if np.random.random() < self.flip_ratio:
            img = self._flip_img(img)
        return img, label


@TRANSFORMS.register_module
class Normalize(object):
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, img, label):
        img = img.astype(np.float32)
        if self.to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - self.mean) / self.std

        return img, label


@TRANSFORMS.register_module
class Pad(object):
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, img):
        h, w = img.shape[:2]
        if self.size:
            pad_h = self.size[1] - h
            pad_w = self.size[0] - w
        elif self.size_divisor:
            pad_h = int(np.ceil(h / self.size_divisor) * self.size_divisor) - h
            pad_w = int(np.ceil(w / self.size_divisor) * self.size_divisor) - w

        assert pad_h >= 0 and pad_w >= 0

        if len(img.shape) == 3:
            padding = np.array([[0, pad_h], [0, pad_w], [0, 0]])
        else:
            padding = np.array([[0, pad_h], [0, pad_w]])

        padded_img = np.pad(
            img,
            padding,
            'constant',
            constant_values=self.pad_val
        )
        return padded_img

    def __call__(self, img, label):
        return self._pad_img(img), label


@TRANSFORMS.register_module
class RandomChannelShift(object):
    def __init__(self, shift_range, ratio=0.5, axis=2):
        self.shift_range = (-shift_range, shift_range) \
            if isinstance(shift_range, (int, float)) else shift_range
        self.ratio = ratio
        self.axis = axis

    def _channel_shift(self, img):
        if np.random.random() < self.ratio:
            intensity = np.random.uniform(*self.shift_range)

            img = np.rollaxis(img, self.axis, 0)
            min_v, max_v = np.min(img), np.max(img)
            channel_imgs = [
                np.clip(img_channel + intensity,
                        min_v,
                        max_v)
                for img_channel in img]
            img = np.stack(channel_imgs, axis=0)
            img = np.rollaxis(img, 0, self.axis + 1)

        return img

    def __call__(self, img, label):
        return self._channel_shift(img), label


@TRANSFORMS.register_module
class RandomAffineTransform(object):
    def __init__(self, rotate_range=None, shear_range=None,
                 shift_range=None, zoom_range=None,
                 rotate_ratio=0.5, shear_ratio=0.5,
                 shift_ratio=0.5, zoom_ratio=0.5,
                 mode='bilinear', border_mode='default', image_value=0):
        self.rotate_range = (-rotate_range, rotate_range) \
            if isinstance(rotate_range, (int, float)) else rotate_range
        self.shift_range = (-shift_range, shift_range) \
            if isinstance(shift_range, (int, float)) else shift_range
        self.shear_range = (-shear_range, shear_range) \
            if isinstance(shear_range, (int, float)) else shear_range
        self.zoom_range = (1 - zoom_range, 1 + zoom_range) \
            if isinstance(zoom_range, (int, float)) else zoom_range

        self.rotate_ratio = rotate_ratio
        self.shift_ratio = shift_ratio
        self.shear_ratio = shear_ratio
        self.zoom_ratio = zoom_ratio

        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value

    def _get_transform_matrix(self, img):
        h, w, c = img.shape
        transform_matrix = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])

        if self.rotate_range:
            if np.random.random() < self.rotate_ratio:
                angle = np.random.uniform(*self.rotate_range)
                center = ((w - 1) * 0.5, (h - 1) * 0.5)
                rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotate_matrix = np.row_stack((rotate_matrix, [0, 0, 1]))
                transform_matrix = np.dot(transform_matrix, rotate_matrix)

        if self.shift_range:
            if np.random.random() < self.shift_ratio:
                tx, ty = np.random.uniform(
                    self.shift_range[0],
                    self.shift_range[1],
                    2)
                tx *= w
                ty *= h

                shift_matrix = np.array([[1, 0, tx],
                                         [0, 1, ty],
                                         [0, 0, 1]])
                transform_matrix = np.dot(transform_matrix, shift_matrix)

        if self.shear_range:
            if np.random.random() < self.shear_ratio:
                shear = np.random.uniform(*self.shear_range)
                shear = np.deg2rad(shear)
                shear_matrix = np.array([[1, -np.sin(shear), 0],
                                         [0, np.cos(shear), 0],
                                         [0, 0, 1]])
                transform_matrix = np.dot(transform_matrix, shear_matrix)

        if self.zoom_range:
            if np.random.random() < self.zoom_ratio:
                zx, zy = np.random.uniform(
                    self.zoom_range[0],
                    self.zoom_range[1],
                    2)

                zoom_matrix = np.array([[zx, 0, 0],
                                        [0, zy, 0],
                                        [0, 0, 1]])
                transform_matrix = np.dot(transform_matrix, zoom_matrix)

        return transform_matrix[:2]

    def _affine_transform(self, img, transform_matrix):
        h, w, _ = img.shape
        affined_img = cv2.warpAffine(
            img,
            M=transform_matrix.astype(np.float64),
            dsize=(w, h),
            flags=self.mode,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.image_value
        )
        return affined_img

    def __call__(self, img, label):
        matrix = self._get_transform_matrix(img)
        return self._affine_transform(img, matrix), label
