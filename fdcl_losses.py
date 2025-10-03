# ===============================
# FILE: src/models/fdcl_losses.py
# ===============================
import tensorflow as tf
from tensorflow.keras import backend as K


def radial_low_high_masks(h, w, cutoff_ratio=0.5):
    y = tf.range(-h//2, h - h//2)
    x = tf.range(-w//2, w - w//2)
    Y, X = tf.meshgrid(y, x, indexing='ij')
    R = tf.sqrt(tf.cast(X*X + Y*Y, tf.float32))
    Rn = R / tf.reduce_max(R)
    low = tf.cast(Rn <= cutoff_ratio, tf.float32)
    high = 1.0 - low
    return tf.signal.ifftshift(low), tf.signal.ifftshift(high)


def fft_split(x, cutoff_ratio=0.5):
    # x: (B, H, W, C)
    x = tf.cast(x, tf.complex64)
    Xf = tf.signal.fft2d(x)
    H = tf.shape(x)[1]; W = tf.shape(x)[2]
    low_mask, high_mask = radial_low_high_masks(H, W, cutoff_ratio)
    low_mask = tf.reshape(low_mask, (1, H, W, 1))
    high_mask = tf.reshape(high_mask, (1, H, W, 1))
    low = tf.signal.ifft2d(Xf * tf.cast(low_mask, tf.complex64))
    high = tf.signal.ifft2d(Xf * tf.cast(high_mask, tf.complex64))
    return tf.math.real(low), tf.math.real(high)


def nt_xent_loss(z1, z2, temperature=0.07):
    # z1, z2: (B, D) positive pairs; negatives formed from batch
    z1 = tf.math.l2_normalize(z1, axis=-1)
    z2 = tf.math.l2_normalize(z2, axis=-1)
    z = tf.concat([z1, z2], axis=0)
    sim = tf.matmul(z, z, transpose_b=True)  # (2B, 2B)
    B2 = tf.shape(z1)[0] * 2
    mask = tf.eye(B2)
    sim = sim - 1e9 * mask  # remove self-sim
    logits = sim / temperature
    labels = tf.concat([tf.range(B2//2, B2), tf.range(0, B2//2)], axis=0)  # positive indices
    labels = tf.one_hot(labels, B2)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)


def fdcl_loss(feature_map, proj_dim=128, temperature=0.07, cutoff=0.5):
    # feature_map: (B, H, W, C)
    low, high = fft_split(feature_map, cutoff_ratio=cutoff)
    GAP = tf.keras.layers.GlobalAveragePooling2D()
    low_z = GAP(low)
    high_z = GAP(high)
    proj = tf.keras.Sequential([
        tf.keras.layers.Dense(proj_dim, activation='relu'),
        tf.keras.layers.Dense(proj_dim)
    ])
    z_low = proj(low_z)
    z_high = proj(high_z)
    # LFCL + HFCL via symmetric NT-Xent
    return nt_xent_loss(z_low, z_high, temperature)
