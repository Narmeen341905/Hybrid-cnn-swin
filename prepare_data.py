# ===============================
# FILE: scripts/prepare_data.py
# ===============================
"""
Usage:
  python scripts/prepare_data.py --source /path/to/dataset --target data/eurosat256 --img_size 256

This script copies images arranged by class folders and (optionally) resizes them once.
"""
import os
import argparse
import shutil
from PIL import Image


def copy_and_resize(src, dst, size):
    os.makedirs(dst, exist_ok=True)
    classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    for c in classes:
        os.makedirs(os.path.join(dst, c), exist_ok=True)
        src_c = os.path.join(src, c)
        for fname in os.listdir(src_c):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            im = Image.open(os.path.join(src_c, fname)).convert("RGB")
            im = im.resize((size, size), Image.BILINEAR)
            im.save(os.path.join(dst, c, os.path.splitext(fname)[0] + ".jpg"), quality=95)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--img_size", type=int, default=256)
    args = ap.parse_args()
    copy_and_resize(args.source, args.target, args.img_size)
