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


def _to_uint8(img):
    if img.dtype == np.uint8: return img
    x = img.astype(np.float32); mn, mx = float(x.min()), float(x.max())
    if mx <= mn: return np.zeros_like(x, dtype=np.uint8)
    return ((x - mn) / (mx - mn) * 255.0).astype(np.uint8)




# 通过饱和度分离组织和背景（饱和度高和白色）
# 去掉噪点，让组织区域更连贯
# 找到轮廓线 cv2.findContours
def build_contour_mask_1024(
    lowres_rgb, # 低分辨率彩色图
    mthresh: int = 41,        # 去噪点
    sthresh: int = 12,        # 分离前景背景
    close: int = 2,           # morphology closing kernel
    min_area_fore: int = 12,  # min area (low-res px) 低分辩像素个数
    min_area_hole: int = 8,  # min hole area (low-res px)
    max_n_holes: int = 8,     # cap holes per region 最多保留孔洞数量
):


    img = _to_uint8(lowres_rgb)
    # 组织区域更有颜色，和白色对比强
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV); sat = hsv[..., 1]
    sat = cv2.medianBlur(sat, int(max(1, mthresh) | 1))
    _, bin_s = cv2.threshold(sat, int(sthresh), 255, cv2.THRESH_BINARY)

    # 填补小缝隙
    if close > 0:
        kernel = np.ones((int(close), int(close)), np.uint8)
        bin_s = cv2.morphologyEx(bin_s, cv2.MORPH_CLOSE, kernel)
    contours, hier = cv2.findContours(bin_s, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(contours) == 0:
        return (bin_s > 0)
    hier = hier[0]  # [Next, Prev, First_Child, Parent]
    fore_ids = [i for i,h in enumerate(hier) if h[3] == -1]
    kept_fore, holes_per_fore = [], []
    for fid in fore_ids:
        a = cv2.contourArea(contours[fid])
        if a <= 0: continue
        # collect children (holes)
        holes = []
        child = hier[fid][2]
        while child != -1:
            holes.append(child)
            child = hier[child][0]
        hole_areas = [cv2.contourArea(contours[h]) for h in holes]
        real_a = a - (np.sum(hole_areas) if hole_areas else 0.0)
        if real_a >= float(min_area_fore):
            kept_fore.append(fid)
            holes_kept = [h for h in holes if cv2.contourArea(contours[h]) > float(min_area_hole)]
            holes_kept = sorted(holes_kept, key=lambda h: cv2.contourArea(contours[h]), reverse=True)[:max_n_holes]
            holes_per_fore.append(holes_kept)
    H, W = bin_s.shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if kept_fore:
        cv2.drawContours(mask, [contours[i] for i in kept_fore], -1, 255, thickness=cv2.FILLED)
    for holes in holes_per_fore:
        if holes:
            cv2.drawContours(mask, [contours[i] for i in holes], -1, 0, thickness=cv2.FILLED)
    return (mask > 0)


# 是否留下patch
# 组织是否较多？是否有边缘？
def patch_keep(mask_bool, x_m, y_m, w_m, h_m):
    H, W = mask_bool.shape[:2]
    x2, y2 = min(W, x_m + w_m), min(H, y_m + h_m)
    if x_m >= x2 or y_m >= y2: return False, {'cov':0.0,'edge':0.0}
    win = mask_bool[y_m:y2, x_m:x2]
    if win.size == 0: return False, {'cov':0.0,'edge':0.0}
    cov = float(win.mean())
    if cov == 0.0:
        edge_ratio = 0.0
    else:
        eroded = cv2.erode(win.astype(np.uint8), np.ones((3,3), np.uint8), 1).astype(bool)
        border = win ^ eroded
        edge_ratio = float(border.mean())
    keep = (cov > 0) or (edge_ratio > 0)
    return keep, {'cov': cov, 'edge': edge_ratio}

# 打分数
# score = 0.6 * 组织覆盖率 + 0.4 * edge
def rank_key(stats: dict, alpha: float = 0.6):
    return alpha*float(stats.get('cov',0.0)) + (1.0-alpha)*float(stats.get('edge',0.0))



def build_contour_mask_1024_explain(
    lowres_rgb,
    mthresh: int = 41,
    sthresh: int = 12,
    close: int = 2,
    min_area_fore: int = 12,
    min_area_hole: int = 8,
    max_n_holes: int = 8,
):
    debug = {}  # 存每一步的中间产物 & 决策日志

    img = _to_uint8(lowres_rgb)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]

    k_med = int(max(1, mthresh) | 1)
    sat_blur = cv2.medianBlur(sat, k_med)
    _, bin_s = cv2.threshold(sat_blur, int(sthresh), 255, cv2.THRESH_BINARY)

    debug['sat'] = sat
    debug['sat_blur'] = sat_blur
    debug['bin_s_thresh'] = (bin_s > 0)

    if close > 0:
        kernel = np.ones((int(close), int(close)), np.uint8)
        bin_s = cv2.morphologyEx(bin_s, cv2.MORPH_CLOSE, kernel)
    debug['bin_s_after_close'] = (bin_s > 0)

    contours, hier = cv2.findContours(bin_s, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    H, W = bin_s.shape
    if hier is None or len(contours) == 0:
        debug['contours'] = []
        return (bin_s > 0), debug

    hier = hier[0]  # [Next, Prev, First_Child, Parent]
    fore_ids = [i for i, h in enumerate(hier) if h[3] == -1]

    kept_fore, holes_per_fore = [], []
    decisions = []  # 记录每个外轮廓的面积、洞面积、是否保留、原因
    for fid in fore_ids:
        a = cv2.contourArea(contours[fid])
        if a <= 0:
            decisions.append({'fid': fid, 'area': a, 'keep': False, 'reason': 'non_positive_area'})
            continue
        # 收集洞
        holes = []
        child = hier[fid][2]
        while child != -1:
            holes.append(child)
            child = hier[child][0]
        hole_areas = [cv2.contourArea(contours[h]) for h in holes]
        real_a = a - (np.sum(hole_areas) if hole_areas else 0.0)

        if real_a >= float(min_area_fore):
            kept_fore.append(fid)
            holes_kept = [h for h in holes if cv2.contourArea(contours[h]) > float(min_area_hole)]
            holes_kept = sorted(holes_kept, key=lambda h: cv2.contourArea(contours[h]), reverse=True)[:max_n_holes]
            holes_per_fore.append(holes_kept)
            decisions.append({
                'fid': fid, 'area': float(a), 'real_area': float(real_a),
                'n_holes': len(holes), 'hole_areas_sum': float(np.sum(hole_areas) if hole_areas else 0.0),
                'keep': True, 'reason': 'ok_area',
            })
        else:
            decisions.append({
                'fid': fid, 'area': float(a), 'real_area': float(real_a),
                'n_holes': len(holes), 'hole_areas_sum': float(np.sum(hole_areas) if hole_areas else 0.0),
                'keep': False, 'reason': 'real_area_below_min_area_fore',
            })

    mask = np.zeros((H, W), dtype=np.uint8)
    if kept_fore:
        cv2.drawContours(mask, [contours[i] for i in kept_fore], -1, 255, thickness=cv2.FILLED)
    for holes in holes_per_fore:
        if holes:
            cv2.drawContours(mask, [contours[i] for i in holes], -1, 0, thickness=cv2.FILLED)

    debug['decisions'] = decisions
    debug['mask_final'] = (mask > 0)
    debug['params'] = dict(
        mthresh=mthresh, sthresh=sthresh, close=close,
        min_area_fore=min_area_fore, min_area_hole=min_area_hole, max_n_holes=max_n_holes
    )
    return (mask > 0), debug

def compute_low_rect_from_hi(x_hi, y_hi, PATCH, downs_hi, downs_low, W_low, H_low, PAD=2):
    # 起点用 floor（整除等价），再对称外扩 PAD
    x_m = (x_hi * downs_hi) // downs_low - PAD
    y_m = (y_hi * downs_hi) // downs_low - PAD
    # 宽高用 ceil（整除等价），再 +2*PAD
    w_m = ((PATCH * downs_hi) + (downs_low - 1)) // downs_low + 2*PAD
    h_m = w_m  # 若你的 PATCH 非正方形，这里分开算

    # 边界裁剪
    x_m = max(0, int(x_m)); y_m = max(0, int(y_m))
    x2  = min(int(W_low), int(x_m + w_m))
    y2  = min(int(H_low), int(y_m + h_m))
    return x_m, y_m, x2, y2, int(x2 - x_m), int(y2 - y_m)