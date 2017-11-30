import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)
from skimage.filters import gaussian
from skimage.color import rgb2gray


# Translate data to an image format
def color_composite(band_1, band_2):
    rgb_arrays = []
    band_3 = band_1 / band_2
    for i in xrange(band_1.shape[0]):
        r = (band_1[i] + abs(band_1[i].min())) / np.max((band_1[i] + abs(band_1[i].min())))
        g = (band_2[i] + abs(band_2[i].min())) / np.max((band_2[i] + abs(band_2[i].min())))
        b = (band_3[i] + abs(band_3[i].min())) / np.max((band_3[i] + abs(band_3[i].min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


def denoise(X, weight, multichannel):
    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])


def smooth(X, sigma):
    return np.asarray([gaussian(item, sigma=sigma) for item in X])


def grayscale(X):
    return np.asarray([rgb2gray(item) for item in X])


def process_input(df):
    smooth_rgb = 0.2
    smooth_gray = 0.05
    weight_rgb = 0.05
    weight_gray = 0.05

    bands_1 = np.stack([np.array(band).reshape(75, 75, 1) for band in df['band_1']], axis=0)
    bands_2 = np.stack([np.array(band).reshape(75, 75, 1) for band in df['band_2']], axis=0)
    bands_3 = (bands_1 + bands_2) / 2

    images = color_composite(bands_1, bands_2)

    bands_1 = smooth(denoise(bands_1, weight_gray, False), smooth_gray)
    bands_2 = smooth(denoise(bands_2, weight_gray, False), smooth_gray)
    bands_3 = smooth(denoise(bands_3, weight_gray, False), smooth_gray)

    X = np.stack([bands_1.squeeze(), bands_2.squeeze(), bands_3.squeeze(),
                  images[:, :, :, 0], images[:, :, :, 1], images[:, :, :, 2]], axis=3).squeeze()
    # X = np.stack([X, images], axis=3).squeeze()
    if 'is_iceberg' in df:
        is_iceberg = df.is_iceberg.reshape(len(df), 1)
        Y = is_iceberg  # .squeeze()
        return X, Y
    else:
        return X, None
