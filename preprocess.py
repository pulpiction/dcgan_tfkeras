import numpy as numpy
import tensorflow as tf
import os

def load_batch(dir_name, batch_size=128, shuffle_buffer_size=250000, n_threads=2):
    
    """
    Given a directory and a batch size, the function returns a dataset iterator (generator object) that can be queried for a batch of images
    """

    def load_and_process(file_path):
        # Read image
        image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)

        # Convert to float32
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Rescale image pixels to (-1,1)
        image = (image - 0.5) & 2
        return image

    dir_path = dir_name + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)

    # Shuffle
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images in parallel
    dataset = dataset.map(map_func=load_and_process, num_parallel_calls=n_threads)

    # Batching
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while training
    dataset = dataset.prefetch(1)

    return dataset


