import tensorflow as tf
import numpy as np
from collections import OrderedDict as OD

''' For MIR '''

def get_next_step_cls(
        current_classifier,
        virtual_classifier,
        latent,
        target
):
    virtual_classifier.set_weights(current_classifier.get_weights())
    virtual_classifier.build(input_shape=(None, *latent.shape[1:]))
    virtual_classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    virtual_classifier.fit(latent, target, latent.shape[0], epochs=1, verbose=0)
    return virtual_classifier



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
