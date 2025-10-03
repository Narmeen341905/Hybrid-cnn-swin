# ===============================
# FILE: src/models/heads.py
# ===============================
import tensorflow as tf
from tensorflow.keras import layers as L


def classification_head(fused_map, n_classes):
    x = L.GlobalAveragePooling2D()(fused_map)
    x = L.Dense(n_classes, activation="softmax")(x)
    return x
