# ===============================
# FILE: src/models/cross_fusion_block.py
# ===============================
import tensorflow as tf
from tensorflow.keras import layers as L


def cross_fusion_block(cnn_feat, swin_feat, out_channels=None, name="cfb"):
    """Concatenate CNN and Swin features -> Conv3x3 -> 1x1 -> residual -> ICA -> split back.
    cnn_feat, swin_feat: (B, H, W, C)
    """
    with tf.name_scope(name):
        # Align shapes if needed
        H1, W1 = tf.shape(cnn_feat)[1], tf.shape(cnn_feat)[2]
        H2, W2 = tf.shape(swin_feat)[1], tf.shape(swin_feat)[2]
        def resize_to(x, H, W):
            return tf.image.resize(x, (H, W), method="bilinear")
        swin_feat = tf.cond(tf.logical_or(tf.not_equal(H1, H2), tf.not_equal(W1, W2)),
                            lambda: resize_to(swin_feat, H1, W1),
                            lambda: swin_feat)
        x = L.Concatenate(axis=-1)([cnn_feat, swin_feat])
        c = x.shape[-1]
        mid = out_channels or c // 2
        y1 = L.Conv2D(mid, 3, padding="same", use_bias=False)(x)
        y1 = L.BatchNormalization()(y1)
        y1 = L.ReLU()(y1)
        # Residual refinement
        y2 = L.Conv2D(mid, 1, padding="same", use_bias=False)(y1)
        y2 = L.BatchNormalization()(y2)
        y2 = L.ReLU()(y2)
        y2 = L.Conv2D(mid, 1, padding="same", use_bias=False)(y2)
        y2 = L.BatchNormalization()(y2)
        y = L.Add()([y1, y2])
        # Improved Channel Attention (ICA)
        w = L.GlobalAveragePooling2D()(y)
        w = L.Dense(mid, activation="relu")(w)
        w = L.Dense(mid, activation="sigmoid")(w)
        w = L.Reshape((1,1,mid))(w)
        y = L.Multiply()([y, w])
        # Split back into two streams (equal split)
        split_ch = mid // 2
        if split_ch == 0:  # safety
            split_ch = mid
        a = L.Conv2D(split_ch, 1, padding="same")(y)
        b = L.Conv2D(split_ch, 1, padding="same")(y)
        # Residual add-back to original streams (resize if needed)
        a = L.Add()([a, L.Conv2D(split_ch, 1, padding="same")(cnn_feat)])
        b = L.Add()([b, L.Conv2D(split_ch, 1, padding="same")(swin_feat)])
        return a, b, y  # return fused map y too
