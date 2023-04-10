import numpy as np
import tensorflow as tf

from distributions import log_normal_diag, log_normal_standard, log_bernoulli

min_epsilon = 1e-6
max_epsilon = 1.-1e-6

def binary_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    #reconstruction_function = nn.BCELoss()
    #reconstruction_function.size_average = False
    reconstruction_function = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    batch_size = x.size(0)

    # - N E_q0 [ ln p(x|z_k) ]
    #bce = reconstruction_function(recon_x, x)
    #bce = -log_Bernoulli(recon_x, x, dim=[0,1,2,3])
    bce = -log_Bernoulli(recon_x, x)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = np.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = np.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = bce + beta * kl

    loss = loss / float(batch_size)
    bce = bce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, bce, kl


def mse_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    batch_size = x.size(0)

    # - N E_q0 [ ln p(x|z_k) ]
    #bce = reconstruction_function(recon_x, x)
    reconstruction_function = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)(x, recon_x)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = np.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = np.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = reconstruction_function + beta * kl

    loss = loss / float(batch_size)
    reconstruction_function = reconstruction_function / float(batch_size)
    kl = kl / float(batch_size)

    return loss, reconstruction_function, kl


def multinomial_loss_function(x_logit, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Computes the cross entropy loss function while summing over batch dimension, not averaged!
    :param x_logit: shape: (batch_size, num_classes * num_channels, pixel_width, pixel_height), real valued logits
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param args: global parameter settings
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    num_classes = 256
    batch_size = x.size(0)

    x_logit = x_logit.view(batch_size, num_classes, args.input_size[0], args.input_size[1], args.input_size[2])

    # make integer class labels
    target = (x * (num_classes-1)).long()

    # - N E_q0 [ ln p(x|z_k) ]
    # sums over batch dimension (and feature dimension)
    # TODO(make sure its F.ce and not ce)
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='sum')(target, x_logit)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = np.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = np.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = ce + beta * kl

    loss = loss / float(batch_size)
    ce = ce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, ce, kl


def binary_loss_array(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta=1.):
    """
    Computes the binary loss without averaging or summing over the batch dimension.
    """

    batch_size = x.size(0)

    # if not summed over batch_dimension
    if len(ldj.size()) > 1:
        ldj = ldj.view(ldj.size(0), -1).sum(-1)

    # TODO: upgrade to newest pytorch version on master branch, there the nn.BCELoss comes with the option
    # reduce, which when set to False, does no sum over batch dimension.
    bce = - log_bernoulli(x.view(batch_size, -1), recon_x.view(batch_size, -1), dim=1)
    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    #  ln q(z_0) - ln p(z_k) ]
    logs = log_q_z0 - log_p_zk

    loss = bce + beta * (logs - ldj)

    return loss


def multinomial_loss_array(x_logit, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Computes the discritezed logistic loss without averaging or summing over the batch dimension.
    """

    num_classes = 256
    batch_size = x.size(0)

    x_logit = x_logit.view(batch_size, num_classes, args.input_size[0], args.input_size[1], args.input_size[2])

    # make integer class labels
    target = (x * (num_classes - 1)).long()

    # - N E_q0 [ ln p(x|z_k) ]
    # computes cross entropy over all dimensions separately:
    ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(target, x_logit)
    # sum over feature dimension
    ce = ce.view(batch_size, -1).sum(dim=1)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k.view(batch_size, -1), dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0.view(batch_size, -1), mean=z_mu.view(batch_size, -1),
                               log_var=z_var.log().view(batch_size, -1), dim=1)

    #  ln q(z_0) - ln p(z_k) ]
    logs = log_q_z0 - log_p_zk

    loss = ce + beta * (logs - ldj)

    return loss



def log_Bernoulli(x_mean, x, average=False, dim=None):
    ''' warning: changed the arg order'''
    probs = tf.clip_by_value(x_mean, clip_value_min=min_epsilon, clip_value_max=max_epsilon)
    log_bernoulli = x * np.log( probs ) + (1. - x ) * np.log( 1. - probs )
    if dim == None:
        dim = np.arange(len(x.shape)).tolist()
    if average:
        return np.mean( log_bernoulli, dim )
    else:
        return np.sum( log_bernoulli, dim )


def calculate_loss(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args, beta=1.):
    """
    Picks the correct loss depending on the input type.
    """

    if args.output_loss == 'bernouilli':
        loss, rec, kl = binary_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)
        bpd = 0.

    elif args.output_loss == 'mse':
        loss, rec, kl = mse_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, beta=beta)
        bpd = 0.

    elif args.output_loss == 'multinomial':
        loss, rec, kl = multinomial_loss_function(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args, beta=beta)
        bpd = loss.item() / (np.prod(args.input_size) * np.log(2.))
    else:
        raise ValueError('Invalid input type for calculate loss: %s.' % args.input_type)

    #TODO(add log_logistic loss for continuous)

    return loss, rec, kl, bpd


def calculate_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args):
    """
    Picks the correct loss depending on the input type.
    """

    if args.input_type == 'binary':
        loss = binary_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj)

    elif args.input_type == 'multinomial':
        loss = multinomial_loss_array(x_mean, x, z_mu, z_var, z_0, z_k, ldj, args)

    else:
        raise ValueError('Invalid input type for calculate loss: %s.' % args.input_type)

    return loss




