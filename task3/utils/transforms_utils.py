import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from loguru import logger


def cv2_to_np(img):
    return img[..., 0]


def np_to_opencv(img):
    """converts a 2D array to a 3 channel opencv grayscale image (make sure image value range is 0-255)"""
    uint_img = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)


def tensor_to_float(tensor):
    return tensor.float()


# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    assert isinstance(image, np.ndarray)

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

    return img_transformed