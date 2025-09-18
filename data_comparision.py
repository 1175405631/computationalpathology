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
import cv2
import numpy as np
import re
from coutour_detection import build_contour_mask_1024_explain, compute_low_rect_from_hi, patch_keep, rank_key, _to_uint8, build_contour_mask_1024
from wsi_io import get_real_tiff, open_pyramid_as_zarr, pick_level_for_target
import tifffile as tiff


def read_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def compare_with_overlap(pathA, pathB, tol=0, out_prefix="cmp"):
    """
    从左上角对齐两张图，比较公共重叠区域像素是否相同，并高亮非重叠区域。
    tol: 容差(0=严格逐像素；5~10 可忽略小插值/量化误差)
    out_prefix: 输出文件名前缀
    """
    A = read_rgb(pathA)
    B = read_rgb(pathB)
    H1, W1 = A.shape[:2]
    H2, W2 = B.shape[:2]

    h = min(H1, H2)
    w = min(W1, W2)

    A_ov = A[:h, :w]
    B_ov = B[:h, :w]

    if tol <= 0:
        equal_mask = np.all(A_ov == B_ov, axis=-1)
    else:
        diff = np.abs(A_ov.astype(np.int16) - B_ov.astype(np.int16))
        equal_mask = np.all(diff <= tol, axis=-1)

    n_total = h * w
    n_equal = int(equal_mask.sum())
    n_diff  = n_total - n_equal
    diff_ratio = n_diff / n_total if n_total else 0.0

    ov_vis = A_ov.copy()
    ov_vis[~equal_mask] = [255, 0, 0]
    ov_blend = cv2.addWeighted(A_ov, 0.6, ov_vis, 0.4, 0)
    cv2.imwrite(f"{out_prefix}_overlap_diff.png", ov_blend)

    Hc, Wc = max(H1, H2), max(W1, W2)
    canvasA = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    canvasB = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    canvasA[:H1, :W1] = A
    canvasB[:H2, :W2] = B

    coverA = np.zeros((Hc, Wc), dtype=np.uint8); coverA[:H1, :W1] = 1
    coverB = np.zeros((Hc, Wc), dtype=np.uint8); coverB[:H2, :W2] = 1
    onlyA = (coverA == 1) & (coverB == 0)
    onlyB = (coverB == 1) & (coverA == 0)

    vis_cover = np.zeros((Hc, Wc, 3), dtype=np.uint8)
    vis_cover[onlyA] = [255, 0, 0]
    vis_cover[onlyB] = [0, 0, 255]
    cv2.imwrite(f"{out_prefix}_non_overlap_map.png", vis_cover)


    stats = {
        "A_shape": (H1, W1),
        "B_shape": (H2, W2),
        "overlap_shape": (h, w),
        "overlap_equal_pixels": n_equal,
        "overlap_diff_pixels": n_diff,
        "overlap_diff_ratio": diff_ratio,   # 重叠区中不同像素占比
        "all_equal_in_overlap": (n_diff == 0),
    }

    # 5) 方便肉眼复核：导出重叠区差异和覆盖差异的叠加图
    # 把覆盖差异叠到较大的画布上显示
    base = canvasA.copy()
    ov_alpha = 0.5
    cover_overlay = cv2.addWeighted(base, 1-ov_alpha, vis_cover, ov_alpha, 0)
    cv2.imwrite(f"{out_prefix}_non_overlap_overlay.png", cover_overlay)

    return stats