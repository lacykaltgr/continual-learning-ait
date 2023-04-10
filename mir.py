# ehhez előbb el kéne olvasni a papert

import tensorflow as tf
import numpy as np

from utils import get_grad_vector, get_future_step_parameters
from loss import calculate_loss


def retrieve_gen_for_gen(args, x, gen, prev_gen, prev_cls):

    grad_vector = get_grad_vector(args, gen.variables, gen.grad_dims)

    virtual_gen = get_future_step_parameters(gen, grad_vector, gen.grad_dims, args.lr)

    _, z_mu, z_var, _, _, _ = prev_gen(x)

    z_new_max = None
    for i in range(args.n_mem):

        with tf.GradientTape(persistent=True) as tape:
            if args.mir_init_prior:
                z_new = prev_gen.prior.sample((z_mu.shape[0], args.z_size)).to(args.device)
            else:
                z_new = prev_gen.reparameterize(z_mu, z_var)

            for j in range(args.mir_iters):
                z_new = tf.Variable(z_new, trainable=True)

                x_new = prev_gen.decode(z_new)

                prev_x_mean, prev_z_mu, prev_z_var, prev_ldj, prev_z0, prev_zk = prev_gen(x_new)
                _, prev_rec, prev_kl, _ = calculate_loss(prev_x_mean, x_new, prev_z_mu,
                        prev_z_var, prev_z0, prev_zk, prev_ldj, args, beta=1)

                virtual_x_mean, virtual_z_mu, virtual_z_var, virtual_ldj, virtual_z0, virtual_zk = virtual_gen(x_new)
                _, virtual_rec, virtual_kl, _ = calculate_loss(virtual_x_mean, x_new, virtual_z_mu,
                        virtual_z_var, virtual_z0, virtual_zk, virtual_ldj, args, beta=1)

                # maximise the interference
                KL = 0
                if args.gen_kl_coeff>0.:
                    KL = virtual_kl - prev_kl

                REC = 0
                if args.gen_rec_coeff>0.:
                    REC = virtual_rec - prev_rec

                # the predictions from the two models should be confident
                ENT = 0
                if args.gen_ent_coeff>0.:
                    y_pre = prev_cls(x_new)
                    ENT = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_pre, logits=y_pre))

                # the new found samples samples should be differnt from each others
                DIV = 0
                if args.gen_div_coeff>0.:
                    for found_z_i in range(i):
                        DIV += tf.reduce_mean(tf.math.squared_difference(
                            z_new,
                            z_new_max[found_z_i * z_new.shape[0]:found_z_i * z_new.shape[0] + z_new.shape[0]])
                            ) / i

                # (NEW) stay on gaussian shell loss:
                SHELL = 0
                if args.gen_shell_coeff>0.:
                    SHELL = tf.reduce_mean(tf.math.squared_difference(
                        tf.norm(z_new, ord=2, axis=1),
                        tf.ones_like(tf.norm(z_new, ord=2, axis=1))*np.sqrt(args.z_size)))

                gain = args.gen_kl_coeff * KL + \
                       args.gen_rec_coeff * REC + \
                       -args.gen_ent_coeff * ENT + \
                       args.gen_div_coeff * DIV + \
                       -args.gen_shell_coeff * SHELL

                z_g = tape.gradient(gain, z_new)
                z_new = (z_new + 1 * z_g).detach()

                if z_new_max is None:
                    z_new_max = tf.identity(z_new)
                else:
                    z_new_max = tf.concat([z_new_max, z_new], axis=0)

            if tf.math.reduce_any(tf.math.is_nan(z_new_max)):
                mir_worked = 0
                mem_x = prev_gen.generate(args.batch_size * args.n_mem)
            else:
                mem_x = prev_gen.decode(z_new_max)
                mir_worked = 1

            return mem_x, mir_worked
