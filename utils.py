import tensorflow as tf
import numpy as np
from collections import OrderedDict as OD

''' For MIR '''

def sum_list(lst):
    total = 0 # initialize the total sum
    for element in lst: # loop through each element in the list
        if isinstance(element, list): # check if the element is a nested list
            total += sum_list(element) # recursively sum the nested list and add to the total
        else:
            total += element # add the element to the total
    return total


def overwrite_grad(parameters, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for layer in parameters:
        if layer.trainable:
            # get the start and end indices of the new gradient vector for this param
            beg = 0 if cnt == 0 else sum_list(grad_dims[:cnt])
            en = sum_list(grad_dims[:cnt + 1])
            # reshape the new gradient vector to match the param shape
            this_grad = tf.reshape(new_grad[beg: en], np.array(layer.trainable_variables).shape)
            # assign the new gradient to the param_grad tensor
            layer.trainable_variables.assign(this_grad)
            # increment the counter
            cnt += 1




def get_grad_vector(parameters, grad_dims):
    """
     gather the gradients in one vector
    """

    grads = tf.constant(0.0, shape=[sum_list(grad_dims)])

    cnt = 0
    for param in parameters:
        if param.trainable:
            with tf.GradientTape() as tape:
                tape.watch(param)
                y = param * 1.0
            grad = tape.gradient(y, param)
            grad = tf.reshape(grad, [-1])
            beg = 0 if cnt == 0 else sum_list(grad_dims[:cnt])
            en = sum_list(grad_dims[:cnt + 1])
            idx = tf.range(beg, en)
            updates = tf.gather(grad, idx) # gather the updates from the gradient tensor
            grads = tf.tensor_scatter_nd_update(grads, tf.expand_dims(idx, axis=1), updates)
        cnt += 1
    return grads


def get_future_step_parameters(this_net, grad_vector, grad_dims, params):
    import classifier
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """

    new_model = classifier.ResNet18(params["n_classes"], nf=20, input_size=params["input_size"])
    new_model.build(input_shape=params["input_size"])
    new_model.set_weights(this_net.get_weights())
    #overwrite_grad(new_model.layers, grad_vector, grad_dims) # assume this function is defined elsewhere
    optimizer = tf.keras.optimizers.SGD(learning_rate=params["lr"]) # create an optimizer with the given learning rate
    optimizer.apply_gradients(zip(grad_vector, new_model.trainable_variables)) # update the new_net parameters with the gradient vector
    return new_model



''' Others '''


def onehot(t, num_classes, device='cpu'):
    """
    convert index tensor into onehot tensor
    :param t: index tensor
    :param num_classes: number of classes
    """
    return tf.zeros(t.size()[0], num_classes).to(device).scatter_(1, t.view(-1, 1), 1)


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
    in PyTorch's cross entropy, targets are expected to be labels
    so to predict probabilities this loss is needed
    suppose q is the target and p is the input
    loss(p, q) = -\sum_i q_i \log p_i
    """
    assert input.shape == target.shape
    input = tf.math.log(tf.clip_by_value(tf.nn.softmax(input, axis=1), 1e-5, 1))
    loss = - tf.reduce_sum(input * target)
    return loss / input.shape[0] if size_average else loss


def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


# def out_mask(t, nc_per_task, n_outputs):
#    # make sure we predict classes within the current task
#    offset1 = int(t * nc_per_task)
#    offset2 = int((t + 1) * nc_per_task)
#    if offset1 > 0:
#        output[:, :offset1].data.fill_(-10e10)
#    if offset2 < self.n_outputs:
#        output[:, offset2:n_outputs].data.fill_(-10e10)



''' LOG '''


def logging_per_task(wandb, log, run, mode, metric, task=0, task_t=0, value=0):
    if 'final' in metric:
        log[run][mode][metric] = value
    else:
        log[run][mode][metric][task_t, task] = value

    if wandb is not None:
        if 'final' in metric:
            wandb.log({mode + metric: value}, step=run)


def print_(log, mode, task):
    to_print = mode + ' ' + str(task) + ' '
    for name, value in log.items():
        # only print acc for now
        if len(value) > 0:
            name_ = name + ' ' * (12 - len(name))
            value = sum(value) / len(value)

            if 'acc' in name or 'gen' in name:
                to_print += '{}\t {:.4f}\t'.format(name_, value)
                # print('{}\t {}\t task {}\t {:.4f}'.format(mode, name_, task, value))

    print(to_print)


def get_logger(names, n_runs=1, n_tasks=None):
    log = OD()
    # log = DD()
    log.print_ = lambda a, b: print_(log, a, b)
    for i in range(n_runs):
        log[i] = {}
        for mode in ['train', 'valid', 'test']:
            log[i][mode] = {}
            for name in names:
                log[i][mode][name] = np.zeros([n_tasks, n_tasks])

            log[i][mode]['final_acc'] = 0.
            log[i][mode]['final_forget'] = 0.

    return log


def get_temp_logger(exp, names):
    log = OD()
    log.print_ = lambda a, b: print_(log, a, b)
    for name in names: log[name] = []
    return log
