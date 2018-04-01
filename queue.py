import tensorflow as tf
import os
import tfrecord


def encode_label(label):
    return int(label)


def get_images_list(file_or_dataset, mode):
"""
The function shows 2 different ways to build that dataset:
- From a root folder, that will have a sub-folder containing images for each class
    '''
    ROOT_FOLDER
        |------ SUB_FOLDER (CLASS 0)
        |           |
        |           | ---- image1.jpg
        |           | ---- image2.jpg
        |           | ---- etc...
        |------- SUB_FOLDER (CLASS 1)
        |           |
        |           | ---- image1.jpg
        |           | ---- image2.jpg
        |           | ---- etc...
    '''
- From a plain text file, that will list all images with their class ID:
    '''
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    '''
Below, there are some parameters that you need to change (Marked 'CHANGE HERE'),

  Args:
    file_or_dataset: a dataset folder or a plain text file.
    mode: 'folder' or 'file'
  Returns:
    return image paths list and its labels list 
"""
    image_paths = []
    labels = []
    if mode == 'file':
        # Reading dataset file
        fid_data = open(file_or_dataset, 'r').read().splitlines()
        for line in fid_data:
            image_paths.append(line.split(' ')[0])
            labels.append(line.split(' ')[1])
        return image_paths, labels
    elif mode == 'folder':
        # An ID will be affected to each sub-folders by alphabetical order
        label = 0
        # Listing the directory
        try:  # Python2
            classes = sorted(os.walk(file_or_dataset).next()[1])
        except Exception:  # Python3
            classes = sorted(os.walk(file_or_dataset).__next__()[1])
        # Listing each sub-directory (the class)
        for sub_dir in classes:
            sub_dir_path = os.path.join(file_or_dataset, sub_dir)
            try:  # Python2
                walk = os.walk(sub_dir_path).next()
            except Exception:  # Python3
                walk = os.walk(sub_dir_path).__next__()
            # Adding each image to the training set
            for sample in walk[2]:  # walk = [root, sub_dirs, files]
                # Only keeps jpeg images
                if sample.endswith('.png'):
                    image_paths.append(os.path.join(sub_dir_path, sample))
                    labels.append(label)
            label += 1
        return image_paths, labels
    else:
        raise Exception("Unknown mode.")


def get_batch_image_and_label(image_paths, labels,
                              image_height, image_width, image_channels,
                              batch_size, shuffle,
                              num_threads=4):
"""
Args:
    image_paths: A list of strings with the image path.
    labels: A list of interger with image label
    image_height: image height
    image_width: image width
    image_channels: image channels
    batch_size: batch size
    suffle: True to shuffle the image and False to Not shuffle the image
    num_threads: number of threads
  Returns:
    Two tensor:
      image_batch: containing a image tensor with the shape (batch_size, image_height, image_width, image_channels)
      lable_batch: containing a label related to iamge with the shape (batch_size, )  
"""
    # Coverting to Tensor
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    # Building a TensorFlow Queue and Shuffling data
    input_queue = tf.train.slice_input_producer([image_paths, labels], shuffle=shuffle)

    # Processing path and string tensor into an image and a label
    label = input_queue[1]
    file_content = tf.read_file(input_queue[0])
    image = tf.image.decode_png(file_content, channels=image_channels)
    # Resize images to a common size
    image = tf.image.resize_images(image, [image_height, image_width])
    # Normalize
    image = image * 1.0 / 127.5 - 1.0
    # Define tensor shape
    image.set_shape([image_height, image_width, image_channels])
    # Collect batches of images before processing
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size,
                                              capacity=batch_size*8, num_threads=num_threads)
    return image_batch, label_batch
