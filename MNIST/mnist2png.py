#!/usr/bin/env/python
import os
import struct
import sys
from array import array
from os import path
import png


def ReadDataset(dataset='training', path= '.'):
    if dataset is 'training':
        images_set_name = os.path.join(path, 'train-images.idx3-ubyte')
        labels_set_name = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is 'testing':
        images_set_name = os.path.join(path, 't10k-images.idx3-ubyte')
        labels_set_name = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'training' or 'testing'")

    fid_images_set = open(images_set_name, 'rb')
    _, size, rows, cols = struct.unpack('>IIII', fid_images_set.read(16))
    images = array('B', fid_images_set.read())
    fid_images_set.close()

    fid_labels_set = open(labels_set_name, 'rb')
    _, size = struct.unpack('>II', fid_labels_set.read(8))
    labels = array('b', fid_labels_set.read())
    fid_labels_set.close()

    return labels, images, size, rows, cols


def WriteDataset(labels, images, size, rows, cols, output_dir):
    output_dir = [path.join(output_dir, str(i)) for i in range(10)]
    for dir in output_dir:
        if not path.exists(dir):
            os.makedirs(dir)

    for (i, label) in enumerate(labels):
        output_filename = path.join(output_dir[label], '{0}_{1}{2}'.format(label, str(i), '.png'))
        print '{0}>>>{1}'.format('writting', output_filename)
        with open(output_filename, 'wb') as fid_out:
            png_writer = png.Writer(cols, rows, greyscale=True)
            data_i = [
                    images[(i*rows*cols + j*cols):(i*rows*cols + (j+1)*cols)]
                    for j in range(rows)
                    ]
            png_writer.write(fid_out, data_i)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print 'using: {0} <input_path> <output_path>'.format(sys.argv[0])
        sys.exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    for dataset in ['training', 'testing']:
        labels, images, size, rows, cols = ReadDataset(dataset, input_path)
        WriteDataset(labels, images, size, rows, cols, path.join(output_path, dataset))
