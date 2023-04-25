import keras
import keras.layers as layer
import tensorflow as tf


# Classifiers
# -----------------------------------------------------------------------------------


class GatedDense(tf.keras.layers.Layer):
    def __init__(self, output_size, activation=None):
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
    def __init__(self, params):
        print("Classifier init")
        super(classifier, self).__init__()

        K = params["cls_hiddens"]
        n_classes = params["n_classes"]

        activation = layer.ReLU()
        self.layer = keras.Sequential([
            layer.Flatten(),
            GatedDense(K, activation=activation),
            layer.Dropout(0.2),
            GatedDense(K, activation=None),
            layer.Dense(n_classes, activation='softmax')
        ])
        self.layer.build(input_shape=(None, 4, 4, 4))

        # get gradient dimension:
        self.grad_dims = self.count_params()

    def call(self, x):
        out = self.layer(x)
        return out

    def count_params(self):
        num_params = []
        for layer in self.layers:
            if layer.trainable:
                layer_params = []
                if hasattr(layer, "layers"): # check if the layer has sub-layers
                    sub_layer_params = layer.count_params() # recursively count the params of sub-layers
                    layer_params.append(sub_layer_params)  # add the params of sub-layers to the layer params
                else:
                    layer_params.append(layer.count_params()) # count the params of the layer
                num_params.append(layer_params)
        return num_params