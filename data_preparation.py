import tensorflow as tf
import torch.utils.data as data
import torchvision.transforms as transforms

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


class RealFakeConditionalDataset(data.Dataset):
    def __init__(self, x_np, y_np, cond_np, transform=transforms.ToTensor()):
        super(RealFakeConditionalDataset, self).__init__()

        self.x = x_np
        self.y = y_np
        self.cond = cond_np
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.x[index]), self.y[index], self.cond[index]

    def __len__(self):
        return len(self.x)




def load_dataset(
        dataset,
        n_classes_first_task=10,
        n_classes_other_task=5,
        img_size=32,
        path=None):
    if (100 - n_classes_first_task)% n_classes_other_task != 0:
        print("Wrong definition of task distribution")
        return None

    if dataset == 'clear-100':
        (X_train, y_train), (X_test, y_test) = load_clear_100(path, img_size)
        n_classes = 100
    elif dataset == 'cifar-100':
        (X_train, y_train), (X_test, y_test) = load_cifar_100()
        n_classes = 100
    elif dataset == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = load_cifar_10()
        n_classes = 10
    else:
        print("Wrong dataset name")
        return None

    datasets_per_task_train = []
    datasets_per_task_test = []
    num_tasks = int((n_classes - n_classes_first_task) / n_classes_other_task) + 1

    for i in range(num_tasks):
        X_task_train = []
        y_task_train = []
        X_task_test = []
        y_task_test = []

        start_class = n_classes_first_task + (i-1) * n_classes_other_task + 1 if i != 0 else 1
        end_class = n_classes_first_task + i * n_classes_other_task

        for j in range(X_train.shape[0]):
            if y_train[j][start_class:end_class].any():
                X_task_train.append(X_train[j])
                y_task_train.append(y_train[j]) #[start_class:end_class])

        datasets_per_task_train.append((X_task_train, y_task_train))

        for j in range(X_test.shape[0]):
            if y_test[j][start_class:end_class].any():
                X_task_test.append(X_test[j])
                y_task_test.append(y_test[j]) #[start_class:end_class])

        datasets_per_task_test.append((X_task_test, y_task_test))

    return datasets_per_task_train, datasets_per_task_test


def load_clear_100(path, img_size=32):
    if path is None: path = "content/drive/MyDrive/clear-dataset/"
    clear_100_features_path = path + "class-names-100.txt"
    X_train, y_train = load_from_tfrecord(path + 'train-clear-100.tfrecord', clear_100_features_path, img_size)
    X_test, y_test = load_from_tfrecord(path + 'test-clear-100.tfrecord', clear_100_features_path, img_size)
    return (X_train, y_train), (X_test, y_test)


def load_cifar_100():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    n_classes = 100
    X_train = (X_train / 127.5) -1
    X_test = (X_test / 127.5) -1
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)
    return (X_train, y_train), (X_test, y_test)

def load_cifar_10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    n_classes = 10
    X_train = (X_train / 127.5) -1
    X_test = (X_test / 127.5) -1
    y_train = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test = tf.keras.utils.to_categorical(y_test, n_classes)
    return (X_train, y_train), (X_test, y_test)

''' Utilities to load and preprocess the data '''

def feature_encode_table(filepath):
    vocab = tf.io.read_file(filepath)
    vocab = tf.strings.split(vocab, sep="\n")
    vocab_size = tf.shape(vocab)[0]

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(vocab, tf.range(vocab_size)),
        default_value = -1
    )
    return table


def encode_task(task, feature_table):
    task_indices = feature_table.lookup(task)
    task_one_hot = tf.one_hot(task_indices, depth=int(feature_table.size()))
    return task_one_hot


def decode_image(image, img_size=32):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.keras.preprocessing.image.smart_resize(image, [img_size, img_size])
    return image


def read_tfrecord_features(example, tfrecord_format, img_size=32):
    example = tf.io.parse_single_example(example, tfrecord_format)
    date = tf.strings.to_number(example['date'], out_type=tf.int32)
    image = decode_image(example['image_raw'], img_size)
    return (image, date)


def read_tfrecord_labels(example, feature_table, tfrecord_format):
    example = tf.io.parse_single_example(example, tfrecord_format)
    task = encode_task(example['task'], feature_table)
    return task


def load_from_tfrecord(record_filepath, feature_filepath, img_size=32):
    tfrecord_format = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "task": tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string)
    }
    dataset = tf.data.TFRecordDataset(record_filepath)
    table = feature_encode_table(feature_filepath)
    X = dataset.map(lambda x: read_tfrecord_features(x, tfrecord_format, img_size))
    y = dataset.map(lambda x: read_tfrecord_labels(x, table, tfrecord_format))
    return (X, y)


''' Download CLEAR-100 dataset and move to tfrecord format for faster loading '''

def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image_string, date, task):
    feature = {
        'date': _bytes_feature(date),
        'task': _bytes_feature(task),
        'image_raw': _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_to_tfrecord(image_paths, tfrecord_file_name):
    with tf.io.TFRecordWriter(f"{tfrecord_file_name}.tfrecord") as writer:
        for path in image_paths:
            date = path.split("/")[-3].encode()
            task = path.split("/")[-2].encode()
            with open(path, "rb") as image:
                image_string = image.read()
                tf_example = image_example(image_string, date, task)
                writer.write(tf_example.SerializeToString())



