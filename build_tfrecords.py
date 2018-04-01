# -*- coding:utf-8 -*-
# !/usr/bin/python
import tfrecords
import queue

def main():
"""
To covert your own dataset to .tfrecords file
"""
    train_images_folder = './MNIST/training'
    test_images_folder = './MNIST/testing'
    train_tfrecords = './MNIST/training.tfrecords'
    test_tfrecords = './MNIST/testing.tfrecords'
    mode = 'folder'
    train_image_paths, train_labels = queue.get_images_list(train_images_folder, mode)
    test_image_paths, test_labels = queue.get_images_list(test_images_folder, mode)
    tfrecords.encode_to_tfrecordes(train_tfrecords, train_image_paths, train_labels)
    tf.records.encode_to_tfrecordes(test_tfrecords, test_image_paths, test_labels)


if __name__ == '__main__':
    main()
