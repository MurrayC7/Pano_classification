# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow import keras
from model import createDenseNet

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data_dir = './data/'
filenames = [os.path.join(data_dir, 'TFcodeX_%d.tfrecord' % (i + 1)) for i in xrange(10)]


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


# data split
def data_split(data, test_size=0.1):
    data_num = len(data[0])
    train_idx = range(data_num)
    # test_idx = []
    # val_idx = []
    test_num = int(data_num * test_size)
    # val_num = int(data_num * val_size)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for i in range(test_num):
        randomIdx = int(np.random.uniform(0, len(train_idx)))
        test_idx = train_idx[randomIdx]
        test_images.append(data[0][test_idx])
        test_labels.append(data[1][test_idx])
        del train_idx[randomIdx]
    for j in train_idx:
        train_images.append(data[0][j])
        train_labels.append(data[1][j])

    return train_images, train_labels, test_images, test_labels


batch = data_load(filenames)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

labels, examples = [], []
with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in xrange(3500):
        id, label, example = sess.run(batch)
        # print "id, labe: ", id, label, example
        example = (np.expand_dims(example, 2))
        labels.append(label)
        examples.append(example)

# plt.figure(figsize=(10, 10))
# for i in xrange(25):
#     plt.subplot(5, 5, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(examples[i], cmap=plt.cm.binary)
#     plt.xlabel(labels[i])
# plt.show()

# data process
train_images, train_labels, test_images, test_labels = data_split([examples, labels])
# print len(examples), len(labels)
# print np.asarray(train_images).shape, np.asarray(train_labels).shape

# Build the model
## setup the layers
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(256, 256)),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
model = createDenseNet(nb_classes=5, img_dim=(256, 256, 1))

## compile the model
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## train the model
model.fit(np.asarray(train_images), train_labels, epochs=5)

## evaluate accuracy
test_loss, test_acc = model.evaluate(np.asarray(test_images), test_labels)
print('Test accuracy:', test_acc)

# Make predictions
single_img = (np.expand_dims(test_images[0], 0))
single_img_label = test_labels[0]
predictions = model.predict(single_img)
results = np.argmax(predictions[0])
print(results, single_img_label)
