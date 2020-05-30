import io
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    from googleapiclient.discovery import build
except Exception:
    build = None


class Parser:
    def __init__(self, is_training: bool, target_size: int):
        self.is_training = is_training
        self.target_size = target_size

    def __call__(self, image_path, scores):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        if self.is_training:
            image = tf.image.random_flip_left_right(image)

        image = tf.image.resize(
            image, (self.target_size, self.target_size),
            method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False)
        image /= 255.0

        return image, scores


def build_dataset(image_paths, scores, is_training, batch_size, target_size):
    parser = Parser(is_training=is_training, target_size=target_size)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, scores))
    if is_training:
        dataset = dataset.shuffle(batch_size, seed=1)

    dataset = dataset.map(parser,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def read_csv(csv_path, image_dir, is_training):
    if build is not None and csv_path.startswith('gs://'):
        gcs_service = build('storage', 'v1')
        bucket = csv_path.split('/')[2]
        csv_path = '/'.join(csv_path.split('/')[3:])
        csv_object = gcs_service.objects().get_media(
            bucket=bucket, object=csv_path).execute()
        csv_path = io.BytesIO(csv_object)

    df = pd.read_csv(csv_path)
    df['image_path'] = df['Image ID'].map(
        lambda x: f'{image_dir}/{x}.jpg')

    if is_training:
        df = df.sample(frac=1, random_state=1)

    image_paths = df['image_path'].values
    scores = df[map(str, range(1, 11))].values.astype(np.float32)

    return image_paths, scores
