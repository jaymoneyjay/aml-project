import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision
from loguru import logger

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
    print(new_dims)

    new_top_left = (box_props['center'][0] - round(new_dims[0] / 2), box_props['center'][1] - round(new_dims[1] / 2))
    new_bottom_right = (new_top_left[0] + new_dims[0], new_top_left[1] + new_dims[1])

    valid_rows = range(0, box_props['mask_dims'][0])
    valid_cols = range(0, box_props['mask_dims'][1])

    # for now, raise an error when the new box would lie outside image
    # TODO: Handle coordinates that are outside image
    if new_top_left[0] not in valid_rows or new_top_left[1] not in valid_cols or new_bottom_right[0] not in valid_rows or new_bottom_right[1] not in valid_cols:
        raise RuntimeError('When creating a new ROI box mask, the coordinates leave the valid coordinate range.')

    new_mask = np.zeros(box_props['mask_dims'], dtype=bool)
    new_mask[new_top_left[0]:new_bottom_right[0], new_top_left[1]:new_bottom_right[1]] = True
    return new_mask

def resize_img(img, width=40, height=30):
    """Resizes 2D Numpy array to given dimensions"""
    # could also try different interpolations such as INTER_CUBIC: https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
    img = img.astype(np.uint8)
    resized = cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return resized

def get_segment_crop(img,tol=0, mask=None):
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

def show_img_batch(batch, title=None):
    batch_frames = batch['frame_cropped']
    batch_labels = batch.get('label_cropped', None)
    logger.debug(batch_frames.shape)
    if batch_labels is not None:
        logger.debug(batch_labels.shape)
    for i in range(batch_frames.shape[0]):
        show_img(batch_frames[i, :, :])
        if batch_labels is not None:
            show_img(batch_labels[i, :, :])

