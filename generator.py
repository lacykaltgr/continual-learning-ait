import tensorflow as tf
from tensorflow import keras
import numpy as np
import math


from keras.layers import Conv2D, GroupNormalization,Dense, Conv2DTranspose

from utils import apply_seq
from constants import _ALPHAS_CUMPROD


class ResBlock(keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            Conv2D(out_channels, 3, strides=(1, 1), padding='same'),
        ]
        self.emb_layers = [
            keras.activations.swish,
            Dense(out_channels),
        ]
        self.out_layers = [
            GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            Conv2D(out_channels, 3, strides=(1, 1), padding='same'),
        ]
        self.skip_connection = (
            Conv2D(out_channels, 3, strides=(1, 1), padding='same') if channels != out_channels else lambda x: x
        )

    def call(self, inputs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        skip_x = self.skip_connection(x)
        ret = skip_x + h
        return ret



class UNetModel(keras.Model):
    def __init__(self, img_height=32, img_width=32, ntype=tf.float32):
        print("UNetModel init")
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.ntype = tf.float32

        self.time_embed = [
            keras.layers.Dense(128),
            keras.activations.swish,
            keras.layers.Dense(128),
        ]
        self.input_blocks = [
            [Conv2D( 32, 3, strides=(1, 1), padding='same')],

            [ResBlock(32, 32)],
            [ResBlock(32, 32)],
            [Conv2D(64, 3, strides=(2, 2), padding='same'),], #downsample

            [ResBlock(32, 64)],
            [ResBlock(64, 64)],
            [Conv2D(128, 3, strides=(2, 2), padding='same'),], #downsample

            [ResBlock(64, 128)],
            [ResBlock(128, 128)],
            [Conv2D(128, 3, strides=(2, 2), padding='same')], #downsample

            [ResBlock(128, 128)],
            [ResBlock(128, 128)],
        ]
        self.middle_block = [
            ResBlock(128, 128),
            ResBlock(128, 128),
        ]
        self.output_blocks = [
            [ResBlock(256, 128)],
            [ResBlock(256, 128)],

            [
                ResBlock(256, 128),
                Conv2DTranspose(128, 2, strides=(2,2)),
                Conv2D(128, 3, strides=(1,1), padding='same')
            ],
            [ResBlock(256, 128)],
            [ResBlock(256, 128)],

            [
                ResBlock(192, 128),
                Conv2DTranspose(128, 2, strides=(2,2)),
                Conv2D(64, 2, strides=(1,1), padding='valid')
            ],
            [ResBlock(192, 64)],
            [ResBlock(128, 64)],

            [
                ResBlock(96, 64),
                Conv2DTranspose(64, 3, strides=(2,2)),
                Conv2D(64, 2, strides=(1,1), padding='valid')
            ],
            [ResBlock(96, 32)],
            [ResBlock(64, 32)],

            [ResBlock(64, 32)],
        ]
        self.out = [
            GroupNormalization(epsilon=1e-5),
            keras.activations.swish,
            Conv2D(8, 3, strides=(1,1), padding='same'),
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
            skip = saved_inputs.pop()
            x = tf.concat([x, skip], axis=-1)
            for layer in b:
                x = apply(x, layer)

        return apply_seq(x, self.out)


    ''' Other functions '''

    def initialize(self, params, input_latent=None, batch_size=64):
        timesteps = np.arange(1, params['num_steps'] + 1) #1, 2, 3, ..., num_steps
        input_lat_noise_t = timesteps[int(len(timesteps) * params["input_latent_strength"])]

        # latent: (batch_size, 32, 32, 8), alphas: (batch_size, num_steps), alphas_prev: (batch_size, num_steps)
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, batch_size, input_latent=input_latent, input_lat_noise_t=input_lat_noise_t
        )
        return latent, alphas, alphas_prev, timesteps

    def get_x_prev(self, x, e_t, a_t, a_prev, temperature):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = x - sqrt_one_minus_at * e_t / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        #noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev

    def get_model_output(self, latent, timestep, batch_size):
        timesteps = tf.convert_to_tensor([timestep], dtype=self.ntype)
        t_emb = self.timestep_embedding(timesteps)
        t_emb = tf.repeat(t_emb, repeats=batch_size, axis=0)
        latent = self.call([latent, t_emb])
        return latent

    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(-math.log(max_period) * np.arange(0, half, dtype="float32") / half)
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=self.ntype)


    def add_noise(self, x, t, noise=None):
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=0)
        batch_size, w, h, c = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        if noise is None:
            noise = tf.random.normal((batch_size, w, h, c), dtype=tf.float32)
        sqrt_alpha_prod = tf.cast(_ALPHAS_CUMPROD[t] ** 0.5, tf.float32)
        sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def get_starting_parameters(self, timesteps, batch_size,  input_latent=None, input_lat_noise_t=None):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if input_latent is None:
            latent = tf.random.normal((batch_size, n_h, n_w, 8))
        else:
            input_latent = tf.cast(input_latent, self.ntype)
            #latent = tf.repeat(input_latent , batch_size , axis=0)
            latent = self.add_noise(input_latent, input_lat_noise_t)
        return latent, alphas, alphas_prev