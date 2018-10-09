from tensorflow import keras


def createDenseNet(nb_classes, img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                   weight_decay=1E-4, verbose=True):
    model_input = keras.layers.Input(shape=img_dim)

    concat_axis = 1 if keras.backend.image_data_format() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = keras.layers.Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same",
                                   name="initial_conv2D",
                                   use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(model_input)

    x = keras.layers.BatchNormalization(axis=concat_axis, gamma_regularizer=keras.regularizers.l2(weight_decay),
                                        beta_regularizer=keras.regularizers.l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                               weight_decay=weight_decay)

    x = keras.layers.Activation('relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(nb_classes, activation='softmax', kernel_regularizer=keras.regularizers.l2(weight_decay),
                           bias_regularizer=keras.regularizers.l2(weight_decay))(
        x)

    densenet = keras.Model(inputs=model_input, outputs=x)

    if verbose:
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet


def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    x = keras.layers.Activation('relu')(input)
    x = keras.layers.Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(x)
    if dropout_rate is not None:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if keras.backend.image_data_format() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = keras.layers.Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter


def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if keras.backend.image_data_format() == "th" else -1

    x = keras.layers.Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(weight_decay))(input)
    if dropout_rate is not None:
        x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(x)

    x = keras.layers.BatchNormalization(axis=concat_axis, gamma_regularizer=keras.regularizers.l2(weight_decay),
                                        beta_regularizer=keras.regularizers.l2(weight_decay))(x)

    return x
