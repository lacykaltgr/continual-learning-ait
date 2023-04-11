import keras
import keras.layers as layer
import numpy as np
import tensorflow as tf


from utils import Reshape


# Classifiers
# -----------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return layer.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=1, bias=False)


class BasicBlock(keras.Model):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = layer.BatchNormalization(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = layer.BatchNormalization(planes)

        self.shortcut = keras.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = keras.Sequential(
                layer.Conv2D(in_planes, self.expansion * planes, kernel_size=1,
                             stride=stride, bias=False),
                layer.BatchNormalization(self.expansion * planes)
            )

    def call(self, x):
        out = layer.ReLU(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = layer.ReLU(out)
        return out


class ResNet(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = tf.keras.layers.Conv2D(nf * 1, kernel_size=3, strides=1, padding='same', input_shape=input_size)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion if input_size[1] in [8, 16, 21, 32, 42] else 640
        self.linear = tf.keras.layers.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return tf.keras.Sequential(layers)

    def return_hidden(self, x):
        out = tf.nn.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = tf.nn.avg_pool2d(out, ksize=4, strides=4, padding='VALID')
        out = tf.reshape(out, (out.shape[0], -1))
        return out

    def call(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20, input_size=(3, 32, 32)):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, input_size)


class GatedDense(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()
        self.activation = activation
        self.h = tf.keras.layers.Dense(output_size)
        self.g = tf.keras.layers.Dense(output_size, activation="sigmoid")

    def call(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation(h)

        g = self.g(x)

        return h * g


''' classifier for GEN and GEN-MIR'''
class classifier(keras.Model):
    def __init__(self, args):
        super(classifier, self).__init__()

        K = args.cls_hiddens
        L = np.prod(args.input_size)
        n_classes = args.n_classes
        self.args = args

        activation = layer.ReLU()
        self.layer = keras.Sequential(
            Reshape([-1]),
            GatedDense(L, K, activation=activation),
            layer.Dropout(p=0.2),
            GatedDense(K, n_classes, activation=None)
        )

        # get gradient dimension:
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

    def call(self, x):
        out = self.layer(x)
        return out