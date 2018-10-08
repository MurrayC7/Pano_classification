# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data_dir = './data/'
filenames = [os.path.join(data_dir, 'TFcodeX_%d.tfrecord' % (i + 1)) for i in xrange(1)]


# data load
def data_load(filenames):
    filename_queue = tf.train.string_input_producer(filenames)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'id': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64),
        'data': tf.FixedLenFeature([256, 256], tf.float32)})
    id = features['id']
    label = features['label']
    image = features['data']

    return id, label, image


batch = data_load(filenames)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

labels, examples = [], []
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(350):
        id, label, example = sess.run(batch)
        # print "id, labe: ", id, label, example
        labels.append(label)
        examples.append(example)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(examples[i], cmap=plt.cm.binary)
    plt.xlabel(labels[i])

# data process

#
