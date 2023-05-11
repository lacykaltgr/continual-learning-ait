import keras
from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, GroupNormalization, Activation, ZeroPadding2D, Dense ,Flatten


class Encoder(keras.Sequential):
    def __init__(self, KERNEL_SIZE = (3, 3), INPUT_SHAPE = (32, 32, 3)):
        super().__init__(
            Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),

            GroupNormalization(epsilon=1e-5),
            Activation("swish"),
            ZeroPadding2D((1, 1)),
            Conv2D(16, 3, strides=(1, 1)),
            ZeroPadding2D((1, 1)),
            Conv2D(8, 1, strides=(1, 1))
        )




class Classifier(keras.Sequential):
    def __init__(self):
        super().__init__(
            Flatten(),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.25),
            Dense(10, activation='softmax')
        )

