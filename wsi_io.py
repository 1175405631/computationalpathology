import os
from pathlib import Path
import zipfile
from glob import glob

import tifffile as tiff, zarr
import numpy as np
import cv2, pandas as pd
from PIL import Image

import torch, torchvision as tv
import h5py

import re
from coutour_detection import build_contour_mask_1024_explain


# unzip the file and return the real tiff
def get_real_tiff(path: str):
    # if file starts with 'PK', it's a ZIP
    with open(path, 'rb') as f:
        sig = f.read(2)
    if sig == b'PK':
        with zipfile.ZipFile(path) as z:
            print('Archive contents:', z.namelist())
            z.extractall(EXTRACT_DIR)
        tiffs = sorted(glob(os.path.join(EXTRACT_DIR, '**', '*.tif*'), recursive=True))
        if not tiffs:
            raise FileNotFoundError('No .tif/.tiff found after extraction.')
        return tiffs[0]
    return path


# zarr: n-dimensional array, like NumPy, but load what you need when you need
# WSIs giant -> zarr
# get the actual array of the pixel data
def _to_zarr_array(znode):
    obj = zarr.open(znode, mode='r')
    # if a single array
    if isinstance(obj, zarr.Array):
        return obj
    # if a group with multi arrays
    if isinstance(obj, zarr.Group):
        keys = list(obj.array_keys())
        if not keys:
            raise ValueError('Zarr Group has no arrays.')
        return obj[keys[0]]
    raise TypeError(f'Unexpected zarr node type: {type(obj)}')


# open the first image in tiff
# collect all levels （pyramid) , 40x --> 20x --> 10x....
def open_pyramid_as_zarr(tf: tiff.TiffFile):

    s0 = tf.series[0]
    arr0 = _to_zarr_array(s0.aszarr())        # level 0 (highest resolution)
    levels = [arr0] + [_to_zarr_array(l.aszarr()) for l in s0.levels]
    # compute downsample factors related to level 0
    # eg: L0 wid = 1000, L1 wid = 500, 1000 / 500 = 2
    downs = [1.0] + [arr0.shape[-2] / lvl.shape[-2] for lvl in levels[1:]]
    return levels, downs

# pick the one closest to my target, 20X here
def pick_level_for_target(downs, target_down=1.0):
    return int(np.argmin([abs(d - target_down) for d in downs]))

# standardize all to RGB unit8 format (H, W, 3)
def ensure_hwc(tile: np.ndarray):
    t = tile
    if t.ndim == 2: # grayscale
        t = np.stack([t]*3, axis=-1)
    elif t.ndim == 3 and t.shape[0] in (3,4) and t.shape[-1] not in (3,4): # (C, H, W)
        t = np.moveaxis(t, 0, -1)  # (C,H,W) -> (H,W,C)
    if t.shape[-1] > 3: # RGBA with alpha
        t = t[..., :3]            # drop alpha
    if t.dtype != np.uint8: # other format
        # best-effort clamp/convert (many WSIs are already uint8)
        t = np.clip(t, 0, 255).astype(np.uint8)
    return t

def save_png(arr: np.ndarray, path: Path):
    Image.fromarray(arr).save(path, format='PNG', compress_level=3)