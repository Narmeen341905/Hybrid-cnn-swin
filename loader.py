# ===============================
# FILE: src/datasets/loader.py
# ===============================
import os
import random
from typing import Tuple, List
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

def _list_images(root: str) -> List[Tuple[str, int]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        d = os.path.join(root, c)
        for fname in os.listdir(d):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((os.path.join(d, fname), class_to_idx[c]))
    return items


def stratified_split(items: List[Tuple[str, int]], train=0.6, val=0.1, test=0.3, seed=42):
    by_cls = {}
    for path, y in items:
        by_cls.setdefault(y, []).append(path)
    rng = random.Random(seed)
    train_set, val_set, test_set = [], [], []
    for y, paths in by_cls.items():
        rng.shuffle(paths)
        n = len(paths)
        n_train = int(n * train)
        n_val = int(n * val)
        cls_train = [(p, y) for p in paths[:n_train]]
        cls_val = [(p, y) for p in paths[n_train:n_train + n_val]]
        cls_test = [(p, y) for p in paths[n_train + n_val:]]
        train_set += cls_train
        val_set += cls_val
        test_set += cls_test
    rng.shuffle(train_set); rng.shuffle(val_set); rng.shuffle(test_set)
    return train_set, val_set, test_set


def _decode_and_resize(path, label, image_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size, method=tf.image.ResizeMethod.BILINEAR)
    # Normalize with ImageNet stats
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img, tf.cast(label, tf.int32)


def make_ds(items, image_size=(256, 256), batch_size=32, shuffle=True, augment=False):
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(min(10000, len(items)))
    ds = ds.map(lambda p, y: _decode_and_resize(p, y, image_size), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def _augment(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.image.random_contrast(img, 0.95, 1.05)
    return img, y


def load_dataset_roots(root: str, img_size=(256, 256), batch=32, seed=42):
    items = _list_images(root)
    train_items, val_items, test_items = stratified_split(items, seed=seed)
    n_classes = len(set([y for _, y in items]))
    return {
        "train": make_ds(train_items, img_size, batch, shuffle=True, augment=True),
        "val": make_ds(val_items, img_size, batch, shuffle=False, augment=False),
        "test": make_ds(test_items, img_size, batch, shuffle=False, augment=False),
        "n_classes": n_classes,
    }
