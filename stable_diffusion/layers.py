import tensorflow as tf
from tensorflow import keras


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D((padding, padding))
        self.conv2d = keras.layers.Conv2D(
            channels, kernel_size, strides=(stride, stride)
        )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)


def apply_seq(x: object, layers: object) -> object:
    for l in layers:
        x = l(x)
    return x