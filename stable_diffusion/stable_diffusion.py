import numpy as np
from tqdm import tqdm
import math

import tensorflow as tf
from tensorflow import keras

from autoencoder_kl import Decoder, Encoder
from diffusion_model import UNetModel

_ALPHAS_CUMPROD, PYTORCH_CKPT_MAPPING = None, None


class StableDiffusion:
    def __init__(self, img_height=1000, img_width=1000, jit_compile=False, download_weights=True):
        self.img_height = img_height
        self.img_width = img_width
        #self.tokenizer = SimpleTokenizer()

        diffusion_model, decoder, encoder = get_models(img_height, img_width, download_weights=download_weights)
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.encoder = encoder

        if jit_compile:
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)
            self.encoder.compile(jit_compile=True)

        self.dtype = tf.float32
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            self.dtype = tf.float16

    def generate(
        self,
        batch_size=1,
        num_steps=25,
        temperature=1,
        seed=None,
    ):
        #TODO: initialize with random noise (latent, timesteps etc)
        timesteps = np.arange(1, num_steps + 1)
        latent_size = 256
        latent = np.random.normal(size=(batch_size, latent_size))
        alphas = np.zeros((num_steps, batch_size, 1, 1), dtype="float32")
        alphas_prev = np.zeros((num_steps, batch_size, 1, 1), dtype="float32")

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f'{index:3d} {timestep:3d}')
            e_t = self.get_model_output(
                latent,
                timestep,
                batch_size
            )
            a_t, a_prev = alphas[index], alphas_prev[index]
            latent, pred_x0 = self.get_x_prev_and_pred_x0(latent, e_t, index, a_t, a_prev, temperature, seed)

        # Decoding stage
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255

        return np.clip(decoded, 0, 255).astype("uint8")

    def get_x_prev_and_pred_x0(self, x, e_t, index, a_t, a_prev, temperature, seed):
        sigma_t = 0
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)

        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * tf.random.normal(x.shape, seed=seed) * temperature
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev, pred_x0


    def get_model_output(
            self,
            latent,
            timestep,
            batch_size,
    ):
        timesteps = np.array([timestep])
        t_emb = self.timestep_embedding(timesteps)
        t_emb = np.repeat(t_emb, batch_size, axis=0)
        latent = self.diffusion_model.predict_on_batch([latent, t_emb])
        return latent


    def timestep_embedding(self, timesteps, dim=320, max_period=10000):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=self.dtype)



    #TODO: nagyon máshogy kell ezt majd
    def load_weights(self, path):
        pt_weights = tf.train.load_checkpoint(path)
        for module_name in ['text_encoder', 'diffusion_model', 'decoder', 'encoder']:
            module_weights = []
            for key, perm in PYTORCH_CKPT_MAPPING[module_name]:
                w = pt_weights.get_tensor(key).numpy()
                if perm is not None:
                    w = np.transpose(w, perm)
                module_weights.append(w)
            getattr(self, module_name).set_weights(module_weights)
            print("Loaded %d weights for %s" % (len(module_weights), module_name))

    #--------------
    # for model with input image/prompt

    def add_noise(self, x , t , noise=None ):
        batch_size,w,h = x.shape[0] , x.shape[1] , x.shape[2]
        if noise is None:
            noise = tf.random.normal((batch_size,w,h,4), dtype=self.dtype)
        sqrt_alpha_prod = _ALPHAS_CUMPROD[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - _ALPHAS_CUMPROD[t]) ** 0.5

        return  sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise

    def get_starting_parameters(self, timesteps, batch_size, seed,  input_image=None, input_img_noise_t=None):
        n_h = self.img_height // 8
        n_w = self.img_width // 8
        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]
        if input_image is None:
            latent = tf.random.normal((batch_size, n_h, n_w, 4), seed=seed)
        else:
            latent = self.encoder(input_image)
            latent = tf.repeat(latent , batch_size , axis=0)
            latent = self.add_noise(latent, input_img_noise_t)
        return latent, alphas, alphas_prev





def get_models(img_height, img_width, download_weights=True):
    n_h = img_height // 8
    n_w = img_width // 8

    # Creation diffusion UNet
    t_emb = keras.layers.Input((320,))
    latent = keras.layers.Input((n_h, n_w, 4))
    unet = UNetModel()
    diffusion_model = keras.models.Model(
        [latent, t_emb], unet([latent, t_emb])
    )

    # Create decoder
    latent = keras.layers.Input((n_h, n_w, 4))
    decoder = Decoder()
    decoder = keras.models.Model(latent, decoder(latent))

    inp_img = keras.layers.Input((img_height, img_width, 3))
    encoder = Encoder()
    encoder = keras.models.Model(inp_img, encoder(inp_img))

    if download_weights:
        diffusion_model_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/diffusion_model.h5",
            file_hash="a5b2eea58365b18b40caee689a2e5d00f4c31dbcb4e1d58a9cf1071f55bbbd3a",
        )
        decoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/decoder.h5",
            file_hash="6d3c5ba91d5cc2b134da881aaa157b2d2adc648e5625560e3ed199561d0e39d5",
        )

        encoder_weights_fpath = keras.utils.get_file(
            origin="https://huggingface.co/divamgupta/stable-diffusion-tensorflow/resolve/main/encoder_newW.h5",
            file_hash="56a2578423c640746c5e90c0a789b9b11481f47497f817e65b44a1a5538af754",
        )

        diffusion_model.load_weights(diffusion_model_weights_fpath)
        decoder.load_weights(decoder_weights_fpath)
        encoder.load_weights(encoder_weights_fpath)

    return diffusion_model, decoder, encoder
