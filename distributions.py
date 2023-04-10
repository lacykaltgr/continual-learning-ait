from __future__ import print_function
import tensorflow as tf

import math
import numpy as np

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.-1e-5

PI = tf.Variable(tf.FloatTensor([math.pi]))
PI.requires_grad = False
if tf.test.is_gpu_available():
    PI = PI.cuda()

# N(x | mu, var) = 1/sqrt{2pi var} exp[-1/(2 var) (x-mean)(x-mean)]
# log N(x| mu, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return np.mean(log_norm, dim)
        else:
            return np.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -(x - mean) * (x - mean)
    log_norm *= np.reciprocal(2.*log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * np.log(2. * PI)

    if reduce:
        if average:
            return np.mean(log_norm, dim)
        else:
            return np.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return np.mean(log_norm, dim)
        else:
            return np.sum(log_norm, dim)
    else:
        return log_norm


def log_bernoulli(x, mean, average=False, reduce=True, dim=None):
    probs = np.clamp(mean, min=MIN_EPSILON, max=MAX_EPSILON)
    log_bern = x * np.log(probs) + (1. - x) * np.log(1. - probs)
    if reduce:
        if average:
            return np.mean(log_bern, dim)
        else:
            return np.sum(log_bern, dim)
    else:
        return log_bern


