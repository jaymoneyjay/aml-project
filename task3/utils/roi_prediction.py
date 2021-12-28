import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger


class Window:
    def __init__(self, x0, y0, width, height):
        self.x = x0
        self.y = y0
        self.width = width
        self.height = height

    def crop(self, img):
        assert self.x + self.width < img.shape[0]
        assert self.y + self.height < img.shape[1]

        return img[
               self.y: self.y + self.height,
               self.x: self.x + self.width
               ]


def generate_windows(img_dims=(112, 112), grid=5, w_dims=(30, 40)):
    w_height, w_width = w_dims
    img_height, img_width = img_dims

    windows = []

    # TODO: use heatmap as prior instead of offsets
    # offset_x = img_width // 2 - grid * 3
    # offset_y = 20

    for j in range(0, img_height, grid):
        for i in range(0, img_width, grid):
            if img_width < i + w_width or img_height < j + w_height:
                continue
            windows.append(Window(i, j, w_width, w_height))
    return windows


def spectral_norm(video, windows, k=5):
    # According to Automatic Mitral Leaflet Tracking in Echocardiography
    # by Outlier Detection in the Low-rank Representation
    # Followint Algorithm 1

    eps = []

    for w in windows:
        M = []

        for i in range(video.shape[2]):
            frame = video[..., i]
            m = w.crop(frame).flatten()
            M.append(m)

        M = np.stack(M, axis=1)
        s = np.linalg.svd(M, compute_uv=False)
        e = np.sum(np.square(s[k:]))
        eps.append(e)

    return np.array(eps)


def gen_heatmap(data, show=False):
    boxes = []
    for sample in data:
        box = sample['box']
        logger.debug('box_shape: {}', box.shape)
        boxes.append(box)

    boxes = np.stack(boxes, axis=0)
    heat_map = np.mean(boxes, axis=0)
    if show:
        plt.imshow(heat_map)
        plt.axis('off')
    return heat_map


def heatmap_norm(data, windows):
    norms = []
    heatmap = gen_heatmap(data)
    for w in windows:
        h = w.crop(heatmap).flatten()
        norms.append(np.sum(np.square(h)))
    return np.array(norms)


def predict_window(norms, weights, windows):
    assert len(norms) == len(weights)
    norms_scaled = []
    for n in norms:
        n_scaled = MinMaxScaler().fit_transform(n.reshape(-1, 1))
        norms_scaled.append(n_scaled)

    n_combined = 0
    for n, w in zip(norms_scaled, weights):
        n_combined += w * n

    l_star = np.argmax(n_combined)
    w_star = windows[l_star]

    return w_star


def get_windows(data):
    frame_shape = data[0]['video'][:, :, 0].shape
    logger.debug('frame_shape: {}', frame_shape)
    return generate_windows(w_dims=frame_shape)


def get_windows_and_heatmap(data):
    windows = get_windows(data)
    heat_norm = heatmap_norm(data, windows)

    return {
        'windows': windows,
        'heatmap': heat_norm,
    }

def get_box_for_video(video, windows, heat_norm, w_dims=(30, 40)):
    s_norm = spectral_norm(video, windows)
    w_star = predict_window([s_norm, heat_norm], [1, 1], windows)

    return w_star
