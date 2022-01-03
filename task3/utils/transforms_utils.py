import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from loguru import logger
import random
from functools import partial
from torchvision.transforms import InterpolationMode
from torchvision import transforms
import torchvision.transforms.functional as tf
from task3.utils.img_utils import adaptive_histogram_equalization


def cv2_to_np(img):
    return img[..., 0]


def np_to_opencv(img):
    """converts a 2D array to a 3 channel opencv grayscale image (make sure image value range is 0-255)"""
    uint_img = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)


def tensor_to_float(tensor):
    return tensor.float()

def to_scaled_tensor(image, mask=None, device='cpu'):
    if mask is None:
        return transforms.ToTensor()(adaptive_histogram_equalization(image, clip_limit=1.25, tile_grid_size=(3, 3))), None
    return transforms.ToTensor()(image), transforms.ToTensor()(mask).bool()

# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, seed=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    random_state = np.random.RandomState(seed)

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)


    is_bool = image.dtype == bool
    image = np_to_opencv(image)  # PREVIOUSLY: image = cv2.cvtColor(image, cv2.CV_8U)
    shape = image.shape
    shape_size = shape[:2]

    alpha *= shape[1]
    sigma *= shape[1]
    alpha_affine *= shape[1]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_CONSTANT)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    img_transformed = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    if is_bool:
        img_transformed = image.astype(bool)

    return img_transformed[:, :, 0]


def functional_transforms(image, cfg=None, mask=None, device='cpu'):
    if cfg is None:
        logger.warning('No transformation settings provided; no transformations will be applied')
        return None

    hist = random.random() > 0.5
    if hist:
        image = adaptive_histogram_equalization(image, clip_limit=1.5, tile_grid_size=(5, 5))

    # Random elastic transform
    elastic = random.random() < cfg['elastic_transform__p']
    seed = np.random.randint(low=0, high=2**32 - 1)
    lambda_elastic = partial(
        elastic_transform,
        alpha=cfg['elastic_transform__alpha'],
        sigma=cfg['elastic_transform__sigma'],
        alpha_affine=cfg['elastic_transform__alpha_affine'],
        seed=seed,
    )
    if elastic:
        image = lambda_elastic(image)

    image = tf.to_pil_image(image)

    # Random horizontal flipping
    hflip = False # random.random() > 0.5
    if hflip:
        image = tf.hflip(image)

    # Random vertical flipping
    vflip = False  # random.random() > 0.5
    if vflip:
        image = tf.vflip(image)

    affine = random.random() > 0.5
    lambda_affine = partial(tf.affine, angle=random.randrange(-15, 15), translate=cfg['random_affine__translate'], scale=cfg['random_affine__scale'], shear=[random.randrange(-15, 15), random.randrange(-15, 15)], interpolation=InterpolationMode.NEAREST)
    if affine:
        image = lambda_affine(img=image)
    """
    perspective = random() > 0.5
    if perspective:
        image = tf.perspective(img=image, startpoints=[], endpoints=[], interpolation=InterpolationMode.NEAREST)
    """
    # TODO: Perspective
    # TODO: ColorJitter?
    # TODO: Random Cropping?

    image = tf.to_tensor(image)

    # Mask-safe transforms
    if mask is not None:
        mask = mask.astype(np.uint8)
        if elastic:
            mask = lambda_elastic(mask)
        mask = tf.to_pil_image(mask)
        if hflip:
            mask = tf.hflip(mask)
        if vflip:
            mask = tf.vflip(mask)
        if affine:
            mask = lambda_affine(img=mask)
        mask = tf.to_tensor(mask).bool()
    return image, mask
