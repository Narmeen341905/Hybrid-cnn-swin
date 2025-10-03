# ===============================
# FILE: src/models/tailored_cnn.py
# ===============================
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers as L


def conv_bn_relu(x, filters, k=3, s=1):
    x = L.Conv2D(filters, k, s, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    return x


def inverted_bottleneck(x, in_ch, expand=4):
    # MobileNetV2-style block as a lightweight local extractor
    hidden = in_ch * expand
    y = L.Conv2D(hidden, 1, padding="same", use_bias=False)(x)
    y = L.BatchNormalization()(y)
    y = L.ReLU()(y)
    y = L.DepthwiseConv2D(3, padding="same", use_bias=False)(y)
    y = L.BatchNormalization()(y)
    y = L.ReLU()(y)
    y = L.Conv2D(in_ch, 1, padding="same", use_bias=False)(y)
    y = L.BatchNormalization()(y)
    return L.ReLU()(L.Add()([x, y]))


def build_tailored_cnn(input_tensor, channels=(32, 64, 128)):
    x = input_tensor
    x = conv_bn_relu(x, channels[0], 3, 1)
    x = L.MaxPool2D(2)(x)
    x = conv_bn_relu(x, channels[1], 3, 1)
    x = L.MaxPool2D(2)(x)
    x = conv_bn_relu(x, channels[2], 3, 1)
    x = L.AveragePooling2D(2)(x)
    # A couple of inverted bottlenecks for richer local patterns
    x = inverted_bottleneck(x, channels[2])
    x = inverted_bottleneck(x, channels[2])
    return x
