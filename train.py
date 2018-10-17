# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.contrib import keras
from model import createDenseNet
from resnet import resnet_v1

import numpy as np
import os

from datautils import data_load, data_split, getDataGenerator, showpic

data_dir = './data/'
filenames = [os.path.join(data_dir, 'TFcodeX_%d.tfrecord' % (i + 1)) for i in range(10)]
check_point_file = "./checkpoint2/pano_resnet_check_point.h5"

batch_size = 16
epoch = 300
numImg = 350 * 10



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))


def train():
    examples, labels = data_load(filenames, numImg)
    # showpic(examples, labels, 25)

    # data process
    train_images, train_labels, val_images, val_labels = data_split([examples, labels])
    # train_labels_oh = keras.utils.to_categorical(train_labels, 5)
    # val_labels_oh = keras.utils.to_categorical(val_labels, 5)
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels) - 1
    val_images = np.asarray(val_images)
    val_labels = np.asarray(val_labels) - 1
    train_datagen = getDataGenerator(train_phase=True)
    train_datagen = train_datagen.flow(train_images, train_labels, batch_size=batch_size)
    validation_datagen = getDataGenerator(train_phase=False)
    validation_datagen = validation_datagen.flow(val_images, val_labels, batch_size=batch_size)

    # Build the model
    ## setup the layers
    model = createDenseNet(nb_classes=5, img_dim=(256, 256, 1), depth=16)
    # model = resnet_v1((256,256,1), nb_classes=5)
#    model = keras.models.Sequential()
#    model.add(keras.layers.Convolution2D(input_shape=(256, 256, 1), filters=8, kernel_size=3,strides=1,padding='same', data_format='channels_last'))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.MaxPooling2D(pool_size=2,strides=2,data_format='channels_last'))
#    model.add(keras.layers.Convolution2D(16, 3, strides=1, padding='same', data_format='channels_last'))
#    model.add(keras.layers.Activation('relu'))
#    model.add(keras.layers.MaxPooling2D(2, 2, data_format='channels_last'))
#    model.add(keras.layers.Dropout(0.5))
#    model.add(keras.layers.Flatten())
#    model.add(keras.layers.Dense(256, activation='relu'))
#    model.add(keras.layers.Dropout(0.5))
#    model.add(keras.layers.Dense(5, activation='softmax'))

    ## compile the model
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    ## train the model
    # model.fit(train_images, train_labels, epochs=15, batch_size=32)
    history = model.fit_generator(generator=train_datagen,
                                  steps_per_epoch=train_images.shape[0] // batch_size,
                                  epochs=epoch,
                                  validation_data=validation_datagen,
                                  validation_steps=val_images.shape[0] // batch_size,
                                  callbacks=[reduce_lr, keras.callbacks.TensorBoard(log_dir='./tmp/log7')],
                                  verbose=1)
    model.save_weights(check_point_file)
    print("We are done, everything seems OK...")


if __name__ == '__main__':
    train()
