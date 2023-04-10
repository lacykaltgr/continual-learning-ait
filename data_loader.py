import tensorflow as tf


class CLDataLoader(object):
    def __init__(self, datasets_per_task, batch_size, train=True):
        bs = batch_size if train else 64

        self.datasets = datasets_per_task
        self.loaders = [
            tf.data.Dataset.from_tensor_slices(x).shuffle(len(x)).batch(bs).prefetch(tf.data.AUTOTUNE)
            for x in self.datasets
        ]

    def __getitem__(self, idx):
        return self.loaders[idx]

    def __len__(self):
        return len(self.loaders)
