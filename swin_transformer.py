# ===============================
# FILE: src/models/swin_transformer.py
# (Lightweight Swin-style backbone for TF Keras)
# ===============================
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as L


def window_partition(x, window_size):
    # x: (B, H, W, C)
    B, H, W, C = tf.unstack(tf.shape(x))
    x = tf.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))  # (B, numWh, numWw, w, w, C)
    windows = tf.reshape(x, (B * (H // window_size) * (W // window_size), window_size, window_size, C))
    return windows


def window_reverse(windows, window_size, H, W):
    # windows: (B*nW, w, w, C)
    BnW = tf.shape(windows)[0]
    C = tf.shape(windows)[-1]
    B = BnW // ((H // window_size) * (W // window_size))
    x = tf.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, C))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (B, H, W, C))
    return x


class WindowAttention(L.Layer):
    def __init__(self, dim, num_heads, window_size, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.qkv = L.Dense(dim * 3, use_bias=False)
        self.proj = L.Dense(dim)
        self.scale = (dim // num_heads) ** -0.5

    def call(self, x):
        # x: (B*nW, w, w, C) -> flatten -> attn
        BnW, w, _, C = tf.unstack(tf.shape(x))
        n = w * w
        x = tf.reshape(x, (BnW, n, C))
        qkv = self.qkv(x)  # (BnW, n, 3C)
        q, k, v = tf.split(qkv, 3, axis=-1)
        q = tf.reshape(q, (BnW, n, self.num_heads, C // self.num_heads))
        k = tf.reshape(k, (BnW, n, self.num_heads, C // self.num_heads))
        v = tf.reshape(v, (BnW, n, self.num_heads, C // self.num_heads))
        q = tf.transpose(q, (0, 2, 1, 3))
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))
        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (BnW, n, C))
        out = self.proj(out)
        out = tf.reshape(out, (BnW, w, w, C))
        return out


class SwinBlock(L.Layer):
    def __init__(self, dim, num_heads=3, window_size=8, shift=False, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift = shift
        self.norm1 = L.LayerNormalization(epsilon=1e-5)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = L.LayerNormalization(epsilon=1e-5)
        self.mlp = tf.keras.Sequential([
            L.Dense(dim * mlp_ratio, activation=tf.nn.gelu),
            L.Dense(dim)
        ])

    def call(self, x):
        H = tf.shape(x)[1]; W = tf.shape(x)[2]
        shortcut = x
        x = self.norm1(x)
        # Shift if needed
        if self.shift:
            shift = self.window_size // 2
            x = tf.roll(x, shift=[-shift, -shift], axis=[1, 2])
        # Partition windows -> attention -> reverse
        windows = window_partition(x, self.window_size)
        windows = self.attn(windows)
        x = window_reverse(windows, self.window_size, H, W)
        # Reverse shift
        if self.shift:
            shift = self.window_size // 2
            x = tf.roll(x, shift=[shift, shift], axis=[1, 2])
        x = L.Add()([shortcut, x])
        # MLP
        y = self.norm2(x)
        y = self.mlp(y)
        x = L.Add()([x, y])
        return x


def patch_embed(x, embed_dim=48, patch=4):
    # Conv patchify
    x = L.Conv2D(embed_dim, kernel_size=patch, strides=patch, padding="same")(x)
    return x


def patch_merge(x, out_dim):
    # Downsample by 2 using strided conv
    x = L.Conv2D(out_dim, kernel_size=2, strides=2, padding="same")(x)
    return x


def build_swin_backbone(input_tensor, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), embed_dim=48, window=8):
    x = patch_embed(input_tensor, embed_dim, patch=4)
    dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
    for stage, (d, h, dim) in enumerate(zip(depths, heads, dims)):
        if stage > 0:
            x = patch_merge(x, dim)
        for i in range(d):
            x = SwinBlock(dim=dim, num_heads=h, window_size=window, shift=(i % 2 == 1))(x)
    return x
