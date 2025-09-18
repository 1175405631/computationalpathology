import os
from pathlib import Path
import zipfile
from glob import glob
import numpy as np
import numpy as np
import tifffile as tiff, zarr
from PIL import Image
import re
from coutour_detection import build_contour_mask_1024_explain, compute_low_rect_from_hi, patch_keep, rank_key, _to_uint8, build_contour_mask_1024
from wsi_io import get_real_tiff, open_pyramid_as_zarr, pick_level_for_target
from data_comparision import compare_with_overlap, read_rgb

import os, pathlib

BASE_DIR = Path.cwd()

# Set up the parameter
# resolution flag
# reso = '20x'
reso = '10x'
drive_path = BASE_DIR / "Input_data"
image_id = '0005f7aaab2800f6170c399693a96917'
image_filename = f'{image_id}.tiff'



tiff_path = os.path.join(drive_path, image_filename)
ZIP_OR_TIFF_PATH = tiff_path

print("Exists:", os.path.exists(tiff_path))
print("Size (MB):", round(os.path.getsize(tiff_path)/1024/1024, 2))



REAL_TIFF_PATH = get_real_tiff(ZIP_OR_TIFF_PATH)
print('Using TIFF:', REAL_TIFF_PATH)

with tiff.TiffFile(REAL_TIFF_PATH) as tf:
    series = tf.series[0]
    low = series.levels[-1].asarray()
    img = Image.fromarray(low)

# 先建文件夹
orig_dir = BASE_DIR / "original_data"
orig_dir.mkdir(parents=True, exist_ok=True)

# 拼接完整路径
out_path = orig_dir / f"original_{image_id}.png"

# 保存图片
img.save(out_path)

print("Saved:", out_path)


with tiff.TiffFile(REAL_TIFF_PATH) as tf:
    print(len(tf.series[0].levels))   # number of pyramid levels
    for lvl in tf.series[0].levels:
        print(lvl.shape)

if reso == '20x':
  # base level: 20X
  hi_level = 0
  PATCH, STRIDE = 1024, 512
  TARGET_DOWN = 1.0

else:
  # 10x level
  hi_level = 1
  PATCH, STRIDE = 512, 256
  TARGET_DOWN = 2.0

with tiff.TiffFile(REAL_TIFF_PATH) as tf:
    levels, downs = open_pyramid_as_zarr(tf)

print('Pyramid downsample factors vs level-0:', [f'{d:.2f}×' for d in downs])

L = pick_level_for_target(downs, TARGET_DOWN)
arrL = levels[L]

# extract H and W
H, W = arrL.shape[-3], arrL.shape[-2]
print(f'Chosen level: {L}  (downsample {downs[L]:.2f}×),  shape≈({H}, {W}, …)')



# choose the lowest reso to build the mask
low_level = len(levels) - 1
arr_low = levels[low_level]
lowres_rgb = np.asarray(arr_low)  # (H_low, W_low, 3) uint8
H_low, W_low = lowres_rgb.shape[:2]



arr0 = levels[hi_level]
H0, W0 = arr0.shape[0], arr0.shape[1]

# Map factor: pixels at hi_level → pixels at low_level
# If your 'downs' is defined as (down from level 0), then:
#   scale_hi_to_low = downs[hi_level] / downs[low_level]
# For hi_level=0 this simplifies to:
scale_hi_to_low = downs[hi_level] / downs[low_level]

# Build the contour mask at the low-res level
contour_mask = build_contour_mask_1024(lowres_rgb)


patches_root = BASE_DIR / "patches"
patches_root.mkdir(parents=True, exist_ok=True)

out_dir = patches_root / f"{image_id}_out_{reso}"
out_dir.mkdir(parents=True, exist_ok=True)


kept = []
for y in range(0, H0 - PATCH + 1, STRIDE):
    for x in range(0, W0 - PATCH + 1, STRIDE):
        # Map hi-res patch → low-res mask window
        # x_m = int(x * scale_hi_to_low)
        # y_m = int(y * scale_hi_to_low)
        # w_m = max(1, int(PATCH * scale_hi_to_low))
        # h_m = max(1, int(PATCH * scale_hi_to_low))

        x_m, y_m, x2, y2, w_m, h_m = compute_low_rect_from_hi(
            x, y, PATCH, downs[hi_level], downs[low_level], W_low, H_low, PAD=2
        )

        keep, stats = patch_keep(contour_mask, x_m, y_m, w_m, h_m)
        if not keep:
            continue

        score = rank_key(stats, alpha=0.6)
        kept.append((score, x, y))

        # Read hi-res tile directly from zarr and save
        tile = np.asarray(arr0[y:y+PATCH, x:x+PATCH, :])
        if tile.shape[:2] != (PATCH, PATCH):
            continue
        Image.fromarray(_to_uint8(tile), 'RGB').save(os.path.join(out_dir, f'p_x{x}_y{y}_s{score:.3f}.png'))

# 上面的部分，只考虑循环到了最后一个PATCH的倍数就停止了，但假如图片的h和w不是PATCH的整数倍，那么就会在最右边和最左边留下一条细缝没有被cover

# 对齐右边缘，强制放一个窗口贴齐最右边
x = W0 - PATCH
for y in range(0, H0 - PATCH + 1, STRIDE):
    x_m, y_m, x2, y2, w_m, h_m = compute_low_rect_from_hi(
        x, y, PATCH, downs[hi_level], downs[low_level], W_low, H_low, PAD=2
    )
    keep, stats = patch_keep(contour_mask, x_m, y_m, w_m, h_m)
    if keep:
        score = rank_key(stats, alpha=0.6)
        kept.append((score, x, y))
        tile = np.asarray(arr0[y:y+PATCH, x:x+PATCH, :])
        if tile.shape[:2] == (PATCH, PATCH):
            Image.fromarray(_to_uint8(tile), 'RGB').save(
                os.path.join(out_dir, f'p_x{x}_y{y}_s{score:.3f}.png')
            )

# 对齐最下边缘
y = H0 - PATCH
for x in range(0, W0 - PATCH + 1, STRIDE):
    x_m, y_m, x2, y2, w_m, h_m = compute_low_rect_from_hi(
        x, y, PATCH, downs[hi_level], downs[low_level], W_low, H_low, PAD=2
    )
    keep, stats = patch_keep(contour_mask, x_m, y_m, w_m, h_m)
    if keep:
        score = rank_key(stats, alpha=0.6)
        kept.append((score, x, y))
        tile = np.asarray(arr0[y:y+PATCH, x:x+PATCH, :])
        if tile.shape[:2] == (PATCH, PATCH):
            Image.fromarray(_to_uint8(tile), 'RGB').save(
                os.path.join(out_dir, f'p_x{x}_y{y}_s{score:.3f}.png')
            )

# 覆盖右下角那个方块
x = W0 - PATCH
y = H0 - PATCH
x_m, y_m, x2, y2, w_m, h_m = compute_low_rect_from_hi(
    x, y, PATCH, downs[hi_level], downs[low_level], W_low, H_low, PAD=2
)
keep, stats = patch_keep(contour_mask, x_m, y_m, w_m, h_m)
if keep:
    score = rank_key(stats, alpha=0.6)
    kept.append((score, x, y))
    tile = np.asarray(arr0[y:y+PATCH, x:x+PATCH, :])
    if tile.shape[:2] == (PATCH, PATCH):
        Image.fromarray(_to_uint8(tile), 'RGB').save(
            os.path.join(out_dir, f'p_x{x}_y{y}_s{score:.3f}.png')
        )

PATCH_DIR = out_dir
pattern = re.compile(r"p_x(\d+)_y(\d+)_s[\d.]+\.png")

patches = []
max_x, max_y = 0, 0

for fname in os.listdir(PATCH_DIR):
    match = pattern.match(fname)
    if not match:
        continue
    x, y = int(match.group(1)), int(match.group(2))
    img = Image.open(os.path.join(PATCH_DIR, fname))
    w, h = img.size
    patches.append((x, y, img))

    max_x = max(max_x, x + w)
    max_y = max(max_y, y + h)

canvas = Image.new("RGB", (max_x, max_y), (255, 255, 255))

for x, y, img in patches:
    canvas.paste(img, (x, y))
