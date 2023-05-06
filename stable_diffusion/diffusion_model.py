import tensorflow as tf
from tensorflow import keras

from stable_diffusion.layers import PaddedConv2D, apply_seq



class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.emb_layers = [
            keras.activations.swish,
            keras.layers.Dense(out_channels),
        ]
        self.out_layers = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret



class Downsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.op = PaddedConv2D(channels, 3, stride=2, padding=1)

    def call(self, x):
        return self.op(x)


class Upsample(keras.layers.Layer):
    def __init__(self, channels):
        super().__init__()
        self.ups = keras.layers.UpSampling2D(size=(2, 2))
        self.conv = PaddedConv2D(channels, 3, padding=1)

    def call(self, x):
        x = self.ups(x)
        return self.conv(x)


class UNetModel(keras.Model):
    def __init__(self):
        print("UNetModel init")
        super().__init__()
        self.time_embed = [
            keras.layers.Dense(1280),
            keras.activations.swish,
            keras.layers.Dense(1280),
        ]
        self.input_blocks = [
            [PaddedConv2D(320, kernel_size=3, padding=1)],
            [ResBlock(320, 320)],
            [ResBlock(320, 320)],
            [Downsample(320)],
            [ResBlock(320, 640)], #, Conv2D(640, kernel_size=8, padding="same")],
            [ResBlock(640, 640)], #, Conv2D(640, kernel_size=8, padding="same")],
            [Downsample(640)],
            #[ResBlock(640, 1280)], #, Conv2D(1280, kernel_size=8, padding="same")],
            #[ResBlock(1280, 1280)], #, Conv2D(1280, kernel_size=8, padding="same")],
            #[Downsample(1280)],
            #[ResBlock(1280, 1280)],
            #[ResBlock(1280, 1280)],
        ]
        self.middle_block = [
            ResBlock(1280, 1280),
            #Conv2D(1280, kernel_size=8, padding="same"),
            ResBlock(1280, 1280),
        ]
        self.output_blocks = [
            #[ResBlock(2560, 1280)],
            #[ResBlock(2560, 1280)],
            #[ResBlock(2560, 1280), Upsample(1280)],
            #[ResBlock(2560, 1280)], #, Conv2D(1280, kernel_size=8, padding="same")],
            #[ResBlock(2560, 1280)], #, Conv2D(1280, kernel_size=8, padding="same")],
            [
                ResBlock(1920, 1280),
                #Conv2D(1280, kernel_size=8, padding="same"),
                Upsample(1280),
            ],
            [ResBlock(1920, 640)], #, Conv2D(640, kernel_size=8, padding="same")],  # 6
            [ResBlock(1280, 640)], #, Conv2D(640, kernel_size=8, padding="same")],
            [
                ResBlock(960, 640),
                #Conv2D(640, kernel_size=8, padding="same"),
                Upsample(640),
            ],
            [ResBlock(960, 320)], #, Conv2D(320, kernel_size=8, padding="same")],
            [ResBlock(640, 320)], #, Conv2D(320, kernel_size=8, padding="same")],
            [ResBlock(640, 320)], #, Conv2D(320, kernel_size=8, padding="same")],
        ]
        self.out = [
            keras.layers.GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            PaddedConv2D(4, kernel_size=3, padding=1),
        ]

    def call(self, inputs):
        x, t_emb = inputs
        emb = apply_seq(t_emb, self.time_embed)

        def apply(x, layer):
            return layer([x, emb]) if isinstance(layer, ResBlock) else layer(x)

        saved_inputs = []
        for b in self.input_blocks:
            for layer in b:
                x = apply(x, layer)
            saved_inputs.append(x)

        for layer in self.middle_block:
            x = apply(x, layer)

        for b in self.output_blocks:
            x = tf.concat([x, saved_inputs.pop()], axis=-1)
            for layer in b:
                x = apply(x, layer)

        return apply_seq(x, self.out)
