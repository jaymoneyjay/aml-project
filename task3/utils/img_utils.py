import numpy as np
import matplotlib.pyplot as plt
import cv2
from loguru import logger
from torch import squeeze


def np_to_opencv(img):
    """converts a 2D array to a 3 channel opencv grayscale image (make sure image value range is 0-255)"""
    uint_img = np.array(img, dtype=np.uint8)
    return cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)


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


def resize_img(img, width=40, height=30):
    """Resizes 2D Numpy array to given dimensions"""
    # could also try different interpolations such as INTER_CUBIC: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    img = img.astype(np.uint8)
    resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return resized


def get_segment_crop(img, tol=0, mask=None):
    """Get image crop based on a Boolean mask, following https://stackoverflow.com/a/53108489"""
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]


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


def show_img(img):
    plt.imshow(img, interpolation=None)
    plt.show()


def img_is_color(img):
    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10),
                    title_fontsize=30, interpolation=None):
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
        cmap = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')

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
    batch_labels = batch.get('label_cropped', pred)
    logger.debug('Shape of batch frames: {}; shape of batch labels {}', batch_frames.shape,
                 batch_labels.shape if hasattr(batch_labels, 'shape') else batch_labels)
    to_plot = []

    if len(batch_frames.shape) == 4:
        # batch of more than 1 element
        for i in range(batch_frames.shape[0]):
            to_plot.append(batch_frames[i, 0, :, :].numpy())
            if batch_labels is not None:
                to_plot.append(batch_labels[i, 0, :, :].numpy())
    elif batch_frames.shape == 3:  # batch size 1
        to_plot = [batch_frames[0, :, :].numpy()]
        if batch_labels is not None:
            to_plot.append(batch_labels[0, :, :].numpy())
    else:
        logger.warning('Could not visualize batch: Invalid batch dimensions.')
        return

    logger.debug(to_plot[0].shape)

    show_image_list(to_plot, num_cols=4, list_titles=list_titles)
