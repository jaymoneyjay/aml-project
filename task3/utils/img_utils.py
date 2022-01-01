import numpy as np
import matplotlib.pyplot as plt
import cv2
from loguru import logger
import task3.utils.utils
import math

from importlib import reload
import sys

reload(sys.modules['task3.utils.utils'])
from task3.utils.utils import get_ith_element_from_dict_of_tensors


def plot_histogram(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    if img.dtype is bool:
        img = img.astype(np.uint8)

    plt.hist(img.ravel(), bins=50, density=True)
    plt.xlabel("pixel values")
    plt.ylabel("relative frequency")
    plt.title("distribution of pixels")
    plt.show()

def asp_ratio(img):
    """
    Returns the aspect ratio of a one channel image: height/width

    :param img:
    :return:
    """
    return round(img.shape[0] / img.shape[1], 2)


def pad_to_dimensions(img, height=112, width=112):
    """
    Pads a 1 channel 2d numpy array to given width and height (added after)

    :param img:
    :param height:
    :param width:
    :return:
    """
    h = img.shape[0]
    w = img.shape[1]

    assert h <= height
    assert w <= width

    y_padding = height - h
    x_padding = width - w

    padded = np.pad(img, ((0, y_padding), (0, x_padding)), mode='constant')
    return padded


def unpad_to_dimensions(img, orig_dims=(112, 112)):
    """
    Removes padding from right and bottom of image (inverse of pad_to_dimensions).

    :param img:
    :param height: target height
    :param width: target width
    :return:
    """

    return img[:orig_dims[0], :orig_dims[1]]


def mask_to_ratio(mask, height=3, width=4):
    """Takes 2D Boolean Numpy mask (Array) and returns corresponding mask with given aspect ratio"""
    box_props = get_box_props(mask)

    # resize box according to aspect ratio
    bigger_dim_idx = np.argmax(box_props['box_dims'])

    asp_ratio = height / width  # h/w
    if bigger_dim_idx == 0:
        new_dims = (box_props['box_dims'][0], round(box_props['box_dims'][0] / asp_ratio))
    else:
        new_dims = (round(box_props['box_dims'][1] * asp_ratio), box_props['box_dims'][1])

    new_top_left = [box_props['center'][0] - round(new_dims[0] / 2), box_props['center'][1] - round(new_dims[1] / 2)]
    new_bottom_right = [new_top_left[0] + new_dims[0], new_top_left[1] + new_dims[1]]

    # check for coordinates outside img dimensions
    y_max = box_props['mask_dims'][0]
    x_max = box_props['mask_dims'][1]
    valid_rows = range(0, y_max)
    valid_cols = range(0, x_max)

    # y coords
    if new_top_left[0] not in valid_rows:
        new_bottom_right[0] += new_top_left[0]
        new_top_left[0] = 0
    elif new_bottom_right[0] not in valid_rows:
        overhang = y_max - new_bottom_right[0]
        new_top_left[0] += overhang
        new_bottom_right[0] = y_max

    # x coords
    if new_top_left[1] not in valid_cols:
        new_bottom_right[1] += new_top_left[1]
        new_top_left[1] = 0
    if new_bottom_right[1] not in valid_cols:
        overhang = x_max - new_bottom_right[0]
        new_top_left[1] += overhang
        new_bottom_right[1] = x_max

    new_mask = np.zeros(box_props['mask_dims'], dtype=bool)
    try:
        new_mask[new_top_left[0]:new_bottom_right[0], new_top_left[1]:new_bottom_right[1]] = True
    except IndexError:
        raise IndexError('When creating a new ROI box mask, the coordinates leave the valid coordinate range. '
                         '--> more sophisticated handling of this is needed in img_utils.py')

    return new_mask


def resize_img(img, width=40, height=30, interpolation=cv2.INTER_AREA):
    """Resizes 2D Numpy array to given dimensions"""
    # could also try different interpolations such as INTER_CUBIC: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    img = img.astype(np.uint8)
    resized = cv2.resize(img, dsize=(width, height), interpolation=interpolation)
    return resized


def get_segment_crop(img, tol=0, mask=None):
    """Get image crop based on a Boolean mask, following https://stackoverflow.com/a/53108489"""
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


def segment_crop_to_full_image(box_props, segment_crop, orig_image, bool_out=False, orig_frame_dims=None):
    """
    Places a segment crop created with get_segment_crop back into the original image.
    :param dict box_props:
    :param segment_crop:
    :param orig_image:
    :param bool bool_out:
    :return updated_image:
    """
    box_dims = box_props['box_dims']
    top_left = box_props['top_left']
    bottom_right = box_props['bottom_right']
    box_h = box_props['box_dims'][0]
    box_w = box_props['box_dims'][1]

    box_asp_ratio = round(box_dims[0] / box_dims[1], 1)
    segment_asp_ratio = round(segment_crop.shape[0] / segment_crop.shape[1], 1)

    try:
        assert box_asp_ratio == segment_asp_ratio
        assert box_props['mask_dims'] == orig_image.shape
    except AssertionError:
        raise AssertionError(
            'box aspect ratio: {} <---> segment_crop aspect ratio: {} '
            '// mask shape {} <---> orig_image shape: {}'.format(box_dims,
                                                                    segment_crop.shape,
                                                                    box_props[
                                                                        'mask_dims'],
                                                                    orig_image.shape))

    upscaled_segment = resize_img(segment_crop, height=box_h, width=box_w, interpolation=cv2.INTER_NEAREST)
    updated_image = orig_image

    try:
        updated_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = upscaled_segment[:, :]
    except IndexError:
        raise IndexError('When upscaling cropped image, index reached outside original image. Time to debug! ;)')

    if bool_out:
        updated_image = updated_image >= 0.5

    if type(orig_frame_dims) is tuple:
        updated_image = unpad_to_dimensions(updated_image, orig_dims=orig_frame_dims)

    return updated_image


def get_box_props(mask):
    """Returns a dict of properties for the ROI box mask"""
    box_props = {}

    # find top left box corner
    for i_row, row in enumerate(mask):
        i_col = np.argmax(row)
        if i_col != 0:
            box_props['top_left'] = (i_row, i_col)
            break

    # find bottom_right box corner
    for i_row, row in enumerate(np.flip(mask)):
        i_col = np.argmax(row)
        if i_col != 0:
            box_props['bottom_right'] = (mask.shape[0] - i_row, mask.shape[1] - i_col)
            break

    # calc more box_props, coords always (y, x) or (height, width) as per numpy (row, col) coordinates
    box_props['box_dims'] = (
        box_props['bottom_right'][0] - box_props['top_left'][0],
        box_props['bottom_right'][1] - box_props['top_left'][1]
    )  # box_dims (height, width)
    box_props['h_to_w_ratio'] = box_props['box_dims'][0] / box_props['box_dims'][1]
    box_props['center'] = (box_props['top_left'][0] + round(box_props['box_dims'][0] / 2),
                           box_props['top_left'][1] + round(box_props['box_dims'][1] / 2))
    box_props['mask_dims'] = mask.shape

    return box_props


def overlay_bw_img(overlay, foundation, alpha=0.3):
    """
    Overlays one image over the other (black/white --> 1 channel images only)

    :param overlay:
    :param foundation:
    :param alpha:
    :return:
    """
    try:
        assert overlay.shape == foundation.shape
    except AssertionError:
        AssertionError('dimension mismatch - overlay.shape: {}, foundation.shape: {}'.format(overlay.shape, foundation.shape))

    blended_img = foundation + alpha * overlay
    blended_img[blended_img > 1] = 1

    return blended_img


def show_img(img):
    plt.imshow(img, interpolation=None)
    plt.show()


def overlay_and_plot_img(overlay, foundation):
    plt.imshow(foundation, interpolation=None)
    plt.imshow(overlay, interpolation=None, alpha=0.3)
    plt.show()


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30, interpolation=None, ignore_grayscale=False):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale. Following https://stackoverflow.com/a/67992521

    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    interpolation: string or None
        any valid imshow() interpolation value (https://matplotlib.org/stable/gallery/images_contours_and_fields/interpolation_methods.html)
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images = len(list_images)
    num_cols = min(num_images, num_cols)
    num_rows = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):
        img = list_images[i]
        title = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) or ignore_grayscale else 'gray')

        list_axes[i].imshow(img, cmap=cmap, interpolation=interpolation)
        list_axes[i].set_title(title, fontsize=title_fontsize)
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


def show_img_batch(batch, list_titles=None, pred=None):
    if type(batch) is not dict:
        logger.warning('Could not visualize batch: No batch dict provided.')
        return

    batch_frames = batch.get('frame_cropped', np.empty(0))  # ugly, ik... ;)
    batch_frames_orig = batch.get('frame_orig', None)
    batch_labels = batch.get('label_cropped', pred)
    batch_box_props = batch.get('box_mask_props', None)
    batch_orig_frame_dims_h = batch.get('orig_frame_dims', None)[0]
    batch_orig_frame_dims_w = batch.get('orig_frame_dims', None)[1]

    logger.debug('Shape of batch frames: {}; shape of batch labels {}', batch_frames.shape,
                 batch_labels.shape if hasattr(batch_labels, 'shape') else batch_labels)

    to_plot = []
    n_cols = 4

    if batch_frames.shape == 3:  # if batch size is 1
        batch_frames = [batch_frames[0, :, :].numpy()]
        if batch_labels is not None:
            batch_labels = [batch_labels[0, :, :].numpy()]
        if batch_frames_orig is not None:
            batch_frames_orig = [batch_frames_orig[0, :, :].numpy()]

    # batch of more than 1 element
    for i in range(batch_frames.shape[0]):
        frame = batch_frames[i, 0, :, :].numpy()
        plot_histogram(frame)
        to_plot.append(frame)

        n_cols = 4
        box_props = get_ith_element_from_dict_of_tensors(i, dictionary=batch_box_props)
        orig_frame_dims = (batch_orig_frame_dims_h[i].item(), batch_orig_frame_dims_w[i].item())
        logger.debug('original frame dims {}', orig_frame_dims)

        if batch_labels is not None:
            n_cols = 3
            current_label = batch_labels[i, 0, :, :].numpy()
            to_plot.append(current_label)
            to_plot.append(overlay_bw_img(current_label, frame, alpha=0.8))

        if batch_frames_orig is not None:
            n_cols = 4
            frame_orig = unpad_to_dimensions(batch_frames_orig[i, 0, :, :].numpy(), orig_dims=orig_frame_dims)

            if batch_labels is not None:
                upscaled_label = segment_crop_to_full_image(box_props, current_label,
                                                            np.zeros(box_props['mask_dims'], dtype=bool),
                                                            orig_frame_dims=orig_frame_dims)
                to_plot.append(overlay_bw_img(upscaled_label, frame_orig, alpha=0.8))

            else:
                to_plot.append(frame_orig)

    figsize = (n_cols * 4.2, math.ceil(len(to_plot) / n_cols) * 4)

    show_image_list(to_plot, num_cols=n_cols, list_titles=list_titles, figsize=figsize, ignore_grayscale=True)
