# ===============================
# FILE: src/eval/evaluate.py
# ===============================
import argparse
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.datasets.loader import load_dataset_roots


def main(args):
    data = load_dataset_roots(args.data_root, img_size=(args.img_h, args.img_w), batch=args.batch)
    model = tf.keras.models.load_model(args.model_path)
    y_true, y_pred = [], []
    for x, y in data["test"]:
        logits = model(x, training=False)
        preds = tf.argmax(logits, axis=-1).numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--img_h", type=int, default=256)
    ap.add_argument("--img_w", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    main(args)
