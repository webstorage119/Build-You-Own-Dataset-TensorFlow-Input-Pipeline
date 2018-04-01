# -*- coding:utf-8 -*-
# !/usr/bin/python
import tensorflow as tf
import os
import skimage.io as io
import numpy as np


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_to_tfrecordes(tfrecordes_filename, image_paths, labels):
"""
Convert your dataset to .tfrecords file
Args:
    tfrecord_filename: the filename of your .tfrecords file
    image_paths: a list of string containg image path
    labels: a list of label containing a related image path
        image_paths, labels = queue.get_images_list(file_or_dataset, mode)  
"""
    if os.path.exists(tfrecordes_filename):
        print "'%s' exists" % tfrecordes_filename
        input_instruction = raw_input("Do you want to remove it?[y/n]")
        while input_instruction not in ['y', 'Y'] and input_instruction not in ['n', 'N']:
            input_instruction = raw_input("One character 'y' or 'Y' for 'Removing it' and 'n' or 'N' for 'Not Removing it'[y/n]")
        if input_instruction in ['y', 'Y']:
            os.remove(tfrecordes_filename)
        else:
            print "EXIT THE PROGRAMME"
            exit(-1)
    writer = tf.python_io.TFRecordWriter(tfrecordes_filename)
    for image_path, label in zip(image_paths, labels):
        print 'Processing the image --> %s' % image_path
        image = io.imread(image_path)
        height, width = image.shape
        image_string = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'channel': _int64_feature(1),  # the mnist is gray scale image channel = 1
            'image_string': _bytes_feature(image_string),
            'label': _int64_feature(label)
        }))
        example_string = example.SerializeToString()
        writer.write(example_string)
    writer.close()
    print "Done"


def decode_from_tfrecords_tf(filename_queue, batch_size,
                             target_height, target_width, target_channel,
                             shuffle_batch=False,
                             ):
  """
  Using TensorFlow and defining the graph to read and batch images from .tfrecords file
  Args:
    filename_queue: A queue of strings with the filenames to read from.
                    filename_queue = tf.train.string_input_producer([trrecords_filename])
    batch_size: batch size
    target_height: the target height of image to setup
    target_width: the target width of image to setup
    shuffle_batch: True for using tf.train.shuffle_batch() and False for using tf.train.batch()
  Returns:
    Two Tensor:
      image_batch: a uint8 Tensor contains a image tensor with the shape 
                  (batch_size, image_height, image_width, image_channels)
      lable_batch: an int32 Tensor contains a label related to iamge with the shape (batch_size, )
  """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'channel': tf.FixedLenFeature([], tf.int64),
            'image_string': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        })
    image = tf.decode_raw(features['image_string'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.reshape(image, [height, width, target_channel])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.image.resize_image_with_crop_or_pad(
        image=image,
        target_height=target_height,
        target_width=target_width)
    if shuffle_batch:
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * batch_size
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4)

    return images, labels


def decode_from_tfrecords_py(tfrecord_filename):
"""
Using Python to read and batch images from .tfrecords file
filename_queue = tf.train.string_input_producer([trrecords_filename])
"""
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_filename)
    for string_recorde in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_recorde)
        height = int(example.features.feature['height'].int64_list.value(0))
        width = int(example.features.feature['width'].int64_list.value(0))
        channel = int(example.features.feature['channel'].int64_list.value(0))
        image_string = example.features.feature['image_string'].bytes_list.value(0)
        label = int(example.features.feature['label'].int64_list.value(0))
        image_1d = np.fromstring(image_string, dtype=np.uint8)
        image = image_1d.reshape((height, width, channel))
        """
        you can use the image and its label here
        """
