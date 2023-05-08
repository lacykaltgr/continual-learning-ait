import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras

from stable_diffusion.autoencoder_kl import Decoder, Encoder
from stable_diffusion.diffusion_model import UNetModel

from stable_diffusion.constants import _ALPHAS_CUMPROD, PYTORCH_CKPT_MAPPING


class StableDiffusion:
    def __init__(self, img_height=64, img_width=64, jit_compile=False, download_weights=False):
        print("StableDiffusion init")
        self.img_height = img_height
        self.img_width = img_width

        diffusion_model, decoder, encoder = get_models(img_height, img_width, download_weights=download_weights)
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.encoder = encoder

        if jit_compile:
            self.diffusion_model.compile(jit_compile=True)
            #self.decoder.compile(jit_compile=True)
            self.encoder.compile(jit_compile=True)

        self.dtype = tf.float32


    def decode(self, latent):
        decoded = self.decoder(latent, training=False)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def initialize(self, params, input_latent=None):
        timesteps = np.arange(1, params['num_steps']+ 1)
        input_lat_noise_t = timesteps[int(len(timesteps)* params["input_latent_strength"])]
        latent, alphas, alphas_prev = self.get_starting_parameters(
            timesteps, params["batch_size"], input_latent=input_latent, input_lat_noise_t=input_lat_noise_t
        )
        timesteps = timesteps[: int(len(timesteps)*params["input_latent_strength"])]
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
        timesteps = tf.convert_to_tensor([timestep], dtype=tf.float32)
        t_emb = self.timestep_embedding(timesteps)
        t_emb = tf.repeat(t_emb, repeats=batch_size, axis=0)
        latent = self.diffusion_model([latent, t_emb])
        return latent


    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=self.dtype)



    # for model with input latent

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
            latent = tf.random.normal((batch_size, n_h, n_w, 4))
        else:
            input_latent = tf.cast(input_latent, self.dtype)
            #latent = tf.repeat(input_latent , batch_size , axis=0)
            latent = self.add_noise(input_latent, input_lat_noise_t)
        return latent, alphas, alphas_prev





def get_models(img_height, img_width, download_weights=True):
    n_h = img_height // 8
    n_w = img_width // 8

    # Creation diffusion UNet
    t_emb = keras.layers.Input((320,))
    latent = keras.layers.Input((n_h, n_w, 4))
    unet = UNetModel()
    diffusion_output = unet.call([latent, t_emb])
    diffusion_model = keras.Model([latent, t_emb], diffusion_output)

    # Create decoder
    #latent = keras.layers.Input((n_h, n_w, 4))
    #decoder = Decoder()
    #decoder = keras.models.Model(latent, decoder(latent))
    decoder = None

    # Create encoder
    inp_img = keras.layers.Input((img_height, img_width, 3), batch_size=64)
    encoder = Encoder()
    encoder = keras.models.Model(inp_img, encoder(inp_img))

    #print(diffusion_model.get_weights())

    if download_weights:
        #diffusion_model_weights_fpath = keras.utils.get_file(
        #    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
        #    file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
        #)
        #decoder_weights_fpath = keras.utils.get_file(
        #    origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
        #    file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
        #)

        encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
            file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
        )

        #diffusion_model.load_weights(diffusion_model_weights_fpath)
        #decoder.load_weights(decoder_weights_fpath)
        encoder.load_weights(encoder_weights_fpath)

    return diffusion_model, decoder, encoder
