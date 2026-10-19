"""Mixed-degradation synthesis, shared by the fixed test sets (tools/build_mixtures.py) and the
training data loader (on the fly).

All functions work on uint8 HxWx3 numpy images, like AdaIR's datasets.

- add_noise:     AdaIR's convention (utils/dataset_utils.py): clip(x + N(0, sigma^2), 0, 255) -> uint8.
- rain_streaks:  the streak layer of a Rain100L pair, max(rainy - clean, 0) (Rain100L is rendered by
                 adding streaks to the clean image, so this recovers them).
- add_streaks:   haze + rain: streaks added on top of a hazy image (rain is near the camera, in front
                 of the haze), clip -> uint8. The streak layer is cropped / resized to the image size.
"""
import numpy as np
from PIL import Image


def add_noise(img, sigma, rng):
    noisy = img.astype(np.float64) + rng.standard_normal(img.shape) * sigma
    return np.clip(noisy, 0, 255).astype(np.uint8)


def rain_streaks(rainy, clean):
    return np.clip(rainy.astype(np.int16) - clean.astype(np.int16), 0, 255).astype(np.uint8)


def fit_layer(layer, h, w, rng=None):
    """Random crop if the layer is big enough, else resize (bilinear) to h x w."""
    lh, lw = layer.shape[:2]
    if lh >= h and lw >= w and rng is not None:
        y = rng.integers(0, lh - h + 1)
        x = rng.integers(0, lw - w + 1)
        return layer[y:y + h, x:x + w]
    return np.array(Image.fromarray(layer).resize((w, h), Image.BILINEAR))


def add_streaks(img, streaks, rng=None):
    s = fit_layer(streaks, img.shape[0], img.shape[1], rng)
    return np.clip(img.astype(np.int16) + s.astype(np.int16), 0, 255).astype(np.uint8)
