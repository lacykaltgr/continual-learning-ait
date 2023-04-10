import tensorflow as tf
import glob
import tensorflow_datasets as tfds


train_clear_10_image_paths = glob.glob("clear-10/train_image_only/labeled_images/*/*/*.jpg")
test_clear_10_image_paths = glob.glob("clear-10/test/labeled_images/*/*/*.jpg")
train_clear_100_image_paths = glob.glob("clear-100/train_image_only/labeled_images/*/*/*.jpg")
test_clear_100_image_paths = glob.glob("clear-100/test/labeled_images/*/*/*.jpg")




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

write_to_tfrecord(train_clear_10_image_paths, 'train-clear-10')
write_to_tfrecord(test_clear_10_image_paths, 'test-clear-10')
write_to_tfrecord(train_clear_100_image_paths, 'train-clear-100')
write_to_tfrecord(test_clear_100_image_paths, 'test-clear-100')




IMAGE_SIZE = [256, 256]


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

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.keras.preprocessing.image.smart_resize(image, IMAGE_SIZE)
    return image


def read_tfrecord_features(example, tfrecord_format):
    example = tf.io.parse_single_example(example, tfrecord_format)
    date = tf.strings.to_number(example['date'], out_type=tf.int32)
    image = decode_image(example['image_raw'])
    return (image, date)

def read_tfrecord_labels(example, feature_table, tfrecord_format):
    example = tf.io.parse_single_example(example, tfrecord_format)
    task = encode_task(example['task'], feature_table)
    return task


def load_dataset(record_filepath, feature_filepath):
    tfrecord_format = {
        "date": tf.io.FixedLenFeature([], tf.string),
        "task": tf.io.FixedLenFeature([], tf.string),
        "image_raw": tf.io.FixedLenFeature([], tf.string)
    }
    dataset = tf.data.TFRecordDataset(record_filepath)
    table = feature_encode_table(feature_filepath)
    X = dataset.map(lambda x: read_tfrecord_features(x, tfrecord_format))
    y = dataset.map(lambda x: read_tfrecord_labels(x, table, tfrecord_format))
    return (X, y)



X_train_clear_10, y_train_clear_10 = load_dataset('train-clear-10.tfrecord', clear_10_features_path)
X_test_clear_10, y_test_clear_10 = load_dataset('train-clear-10.tfrecord', clear_10_features_path)
X_train_clear_100, y_train_clear_100 = load_dataset('train-clear-10.tfrecord', clear_10_features_path)
X_test_clear_100, y_test_clear_100 = load_dataset('train-clear-10.tfrecord', clear_10_features_path)