# -*- coding: UTF-8 -*-
from tensorflow.contrib import keras

import numpy as np

from model import createDenseNet
from datautils import data_load, getDataGenerator

checkpoint_path = './checkpoint/pano_densenet_check_point.h5'
testImg_num = 170
batch_size = 10


def model_test(filename):
    model = createDenseNet(nb_classes=5, img_dim=(256, 256, 1), depth=16)
    model.load_weights(checkpoint_path)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # data process
    test_images, test_labels = data_load(filename, testImg_num)

    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels) - 1
    test_datagen = getDataGenerator(train_phase=False)
    test_datagen = test_datagen.flow(test_images, test_labels, batch_size=batch_size)

    ## etestuate accuracy
    # test_loss, test_acc = model.ev(test_images, test_labels)
    test_loss, test_acc = model.evaluate_generator(test_datagen,
                                          steps=test_images.shape[0] // batch_size)
    print('Model test accuracy = %.2f' % (test_acc))
    # Make predictions
    label = []
    for i in xrange(testImg_num):
        single_img = (np.expand_dims(test_images[i], 0))
        single_img_label = test_labels[i]
        predictions = model.predict(single_img)
        results = np.argmax(predictions[0])
        # print(predictions, results + 1, single_img_label + 1)
        label.append(results + 1)

    return label


def main():
    label = model_test(['TFcodeX_test.tfrecord'])


if __name__ == '__main__':
    main()
