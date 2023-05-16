import tensorflow as tf
import numpy as np
import torch

'''FOR DIFFUSION'''
class vpsde():
    def __init__(self):
        self.beta_0 = 0.1
        self.beta_1 = 20.
        self.s = 0.008
        self.f_0 = np.cos(self.s / (1. + self.s) * np.pi / 2.) ** 2

    @property
    def T(self):
        return 1

    def compute_tau(self, std_wve_t):
        tau = -self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2. * (self.beta_1 - self.beta_0) * torch.log(1. + std_wve_t ** 2))
        tau /= self.beta_1 - self.beta_0
        return tau

    def marginal_prob(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def transform_unnormalized_wve_to_normalized_vp(self, t, std_out=False):
        tau = self.compute_tau(t)
        mean_vp_tau, std_vp_tau = self.marginal_prob(tau)
        if std_out:
            return mean_vp_tau, std_vp_tau, tau
        return mean_vp_tau, tau

    def compute_t_cos_from_t_lin(self, t_lin):
        sqrt_alpha_t_bar = torch.exp(-0.25 * t_lin ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t_lin * self.beta_0)
        time = torch.arccos(np.sqrt(self.f_0) * sqrt_alpha_t_bar)
        t_cos = self.T * ((1. + self.s) * 2. / np.pi * time - self.s)
        return t_cos

    def get_diffusion_time(self, batch_size, batch_device, t_min=1e-5, importance_sampling=True):
        if importance_sampling:
            Z = self.normalizing_constant(t_min)
            u = torch.rand(batch_size, device=batch_device)
            return (-self.beta_0 + torch.sqrt(self.beta_0 ** 2 + 2 * (self.beta_1 - self.beta_0) *
                                              torch.log(1. + torch.exp(Z * u + self.antiderivative(t_min))))) / (self.beta_1 - self.beta_0), Z.detach()
        else:
            return torch.rand(batch_size, device=batch_device) * (self.T - t_min) + t_min, 1

    def antiderivative(self, t, stabilizing_constant=0.):
        if isinstance(t, float) or isinstance(t, int):
            t = torch.tensor(t).float()
        return torch.log(1. - torch.exp(- self.integral_beta(t)) + stabilizing_constant) + self.integral_beta(t)

    def normalizing_constant(self, t_min):
        return self.antiderivative(self.T) - self.antiderivative(t_min)

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_1 - self.beta_0) + t * self.beta_0

def get_one_hot_predictions(mem_pred):
    maximum = np.argmax(mem_pred, axis=1)
    num_classes = mem_pred.shape[1]
    mem_true = np.zeros_like(mem_pred)
    mem_true[np.arange(len(maximum)), maximum] = 1
    return mem_true


''' For MIR '''

def get_next_step_cls(
        current_classifier,
        virtual_classifier,
        sample,
        target,
        batch_size=256,
):
    virtual_classifier.set_weights(current_classifier.get_weights())
    virtual_classifier.build(input_shape=(None, *sample.shape[1:]))
    virtual_classifier.compile(optimizer="adam", loss="categorical_crossentropy")
    virtual_classifier.fit(sample, target, batch_size=batch_size, epochs=1, verbose=0)
    return virtual_classifier

def get_one_hot_predictions(mem_pred):
    maximum = np.argmax(mem_pred, axis=1)
    num_classes = mem_pred.shape[1]
    mem_true = np.zeros_like(mem_pred)
    mem_true[np.arange(len(maximum)), maximum] = 1
    return mem_true



''' Losses '''

def distillation_KL_loss(y, teacher_scores, T, scale=1, reduction='batchmean'):
    """Computes the distillation loss (cross-entropy).
       xentropy(y, t) = kl_div(y, t) + entropy(t)
       entropy(t) does not contribute to gradient wrt y, so we skip that.
       Thus, loss value is slightly different, but gradients are correct.
       \delta_y{xentropy(y, t)} = \delta_y{kl_div(y, t)}.
       scale is required as kl_div normalizes by nelements and not batch size.
    """
    log_softmax_y = tf.nn.log_softmax(y / T, axis=1)
    softmax_teacher_scores = tf.nn.softmax(teacher_scores / T, axis=1)
    return tf.keras.losses.KLD(softmax_teacher_scores, log_softmax_y, reduction=reduction) * scale


def naive_cross_entropy_loss(input, target, size_average=True):
    """
    cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.shape == target.shape
    input = tf.math.log(tf.clip_by_value(tf.nn.softmax(input, axis=1), 1e-5, 1))
    loss = - tf.reduce_sum(input * target)
    return loss / input.shape[0] if size_average else loss


''' LOG '''
