import torch
from guided_diffusion.script_util import create_classifier
import os
import numpy as np
from utils import vpsde
import pickle

class Generator:
    def __init__(self, img_resolution=32, device='cuda:0', encoder_path=None, scorenet_path=None):
        self.image_resolution = img_resolution
        self.device = device
        self.vpsde = vpsde()
        self.discriminator = self.load_discriminator(None)
        self.encoder = self.load_encoder(encoder_path)
        self.net = self.load_score_network(scorenet_f=scorenet_path)

    def pipeline(self):
        def evaluate(perturbed_inputs, timesteps=None, condition=None):
            adm_features = self.encoder(perturbed_inputs, timesteps=timesteps, feature=True)
            prediction = self.discriminator(adm_features, timesteps, sigmoid=True, condition=condition).view(-1)
            return prediction
        return evaluate

    def get_grad_log_ratio(self, unnormalized_input, std_wve_t, time_min, time_max, class_labels, log=False):
        mean_vp_tau, tau = self.vpsde.transform_unnormalized_wve_to_normalized_vp(std_wve_t) ## VP pretrained classifier
        if tau.min() > time_max or tau.min() < time_min:
            if log:
                return torch.zeros_like(unnormalized_input), 10000000. * torch.ones(unnormalized_input.shape[0], device=unnormalized_input.device)
            return torch.zeros_like(unnormalized_input)
        else:
            input = mean_vp_tau[:,None,None,None] * unnormalized_input
        with torch.enable_grad():
            x_ = input.float().clone().detach().requires_grad_()
            tau = torch.ones(input.shape[0], device=tau.device) * tau
            log_ratio = self.get_log_ratio(x_, tau, class_labels)
            discriminator_guidance_score = torch.autograd.grad(outputs=log_ratio.sum(), inputs=x_, retain_graph=False)[0]
            discriminator_guidance_score *= -((std_wve_t[:,None,None,None] ** 2) * mean_vp_tau[:,None,None,None])
        if log:
            return discriminator_guidance_score, log_ratio
        return discriminator_guidance_score

    def get_log_ratio(self, input, time, class_labels):
        logits = self.pipeline()(input, timesteps=time, condition=class_labels)
        prediction = torch.clip(logits, 1e-5, 1. - 1e-5)
        log_ratio = torch.log(prediction / (1. - prediction))
        return log_ratio

    def load_encoder(self, ckpt_path, eval=True):
        encoder_args = dict(
            image_size=self.image_resolution,
            classifier_use_fp16=False,
            classifier_width=128,
            classifier_depth=4 if self.image_resolution in [64, 32] else 2,
            classifier_attention_resolutions="32,16,8",
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention",
            out_channels=1000,
        )
        encoder = create_classifier(**encoder_args)
        encoder.to(self.device)
        if ckpt_path is not None:
            ckpt_path = os.getcwd() + ckpt_path
            encoder_state = torch.load(ckpt_path, map_location="cpu")
            encoder.load_state_dict(encoder_state)
        if eval:
            encoder.eval()
        return encoder

    def load_discriminator(self, ckpt_path, eval=False, channel=512):
        discriminator_args = dict(
            image_size=8,
            classifier_use_fp16=False,
            classifier_width=128,
            classifier_depth=2,
            classifier_attention_resolutions="32,16,8",
            classifier_use_scale_shift_norm=True,
            classifier_resblock_updown=True,
            classifier_pool="attention",
            out_channels=1,
            in_channels=channel
        )
        discriminator = create_classifier(**discriminator_args)
        discriminator.to(self.device)
        if ckpt_path is not None:
            ckpt_path = os.getcwd() + ckpt_path
            discriminator_state = torch.load(ckpt_path, map_location="cpu")
            discriminator.load_state_dict(discriminator_state)
        if eval:
            discriminator.eval()
        return discriminator

    def load_score_network(self, scorenet_f=None):
        with open(scorenet_f, 'rb') as f:
            scorenet = pickle.load(f)['ema'].to(self.device)
        return scorenet

    def sample(
            self, boosting, time_min, time_max, dg_weight_1st_order, dg_weight_2nd_order,
            latents, class_labels=None, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    ):
        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Settings for boosting
        S_churn_manual = 4.
        S_noise_manual = 1.000
        period = 5
        period_weight = 2
        log_ratio = torch.tensor([np.inf] * latents.shape[0], device=latents.device)
        S_churn_vec = torch.tensor([S_churn] * latents.shape[0], device=latents.device)
        S_churn_max = torch.tensor([np.sqrt(2) - 1] * latents.shape[0], device=latents.device)
        S_noise_vec = torch.tensor([S_noise] * latents.shape[0], device=latents.device)

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next
            S_churn_vec_ = S_churn_vec.clone()
            S_noise_vec_ = S_noise_vec.clone()

            if i % period == 0:
                if boosting:
                    S_churn_vec_[log_ratio < 0.] = S_churn_manual
                    S_noise_vec_[log_ratio < 0.] = S_noise_manual

            # Increase noise temporarily.
            # gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            gamma_vec = torch.minimum(S_churn_vec_ / num_steps, S_churn_max) if S_min <= t_cur <= S_max else torch.zeros_like(S_churn_vec_)
            t_hat = self.net.round_sigma(t_cur + gamma_vec * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt()[:, None, None, None] * S_noise_vec_[:, None, None,None] * torch.randn_like(x_cur)
            #x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            denoised = self.net(x_hat, t_hat, class_labels).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat[:, None, None, None]
            # DG correction
            if dg_weight_1st_order != 0.:
                discriminator_guidance, log_ratio = self.get_grad_log_ratio(x_hat, t_hat, time_min, time_max, class_labels, log=True)
                if boosting:
                    if i % period_weight == 0:
                        discriminator_guidance[log_ratio < 0.] *= 2.
                d_cur += dg_weight_1st_order * (discriminator_guidance / t_hat[:, None, None, None])
            x_next = x_hat + (t_next - t_hat)[:, None, None, None] * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.net(x_next, t_next, class_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                ##DG correction
                if dg_weight_2nd_order != 0.:
                    discriminator_guidance = self.get_grad_log_ratio(x_next, t_next, time_min, time_max, class_labels, log=False)
                    d_prime += dg_weight_2nd_order * (discriminator_guidance / t_next)
                x_next = x_hat + (t_next - t_hat)[:, None, None, None] * (0.5 * d_cur + 0.5 * d_prime)

        return x_next