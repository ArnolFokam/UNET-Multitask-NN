from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *


def bottleneck_layer(x,
                     filter_size,
                     num_filters=3,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal',
                     kernel_regularizer=l2(0.001)):
    x = Conv2D(
        filter_size,
        num_filters,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        padding=padding,
        use_bias=False)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    return x


def bottleneck_resnet_layer(x,
                            filter_size,
                            num_filters=3,
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal'):
    assert isinstance(num_filters, int)

    # first conv block
    x = Conv2D(filter_size,
               num_filters,
               padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # second conv block
    x = Conv2D(filter_size,
               num_filters,
               padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)

    return x


def bottleneck_dense_layer(x,
                           filter_size,
                           num_filters=(1, 3),
                           activation='relu',
                           padding='same',
                           kernel_initializer='he_normal',
                           dropout=None):
    assert len(num_filters) == 2

    # first conv block
    x = Conv2D(filter_size,
               num_filters[1],
               padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if isinstance(dropout, float):
        x = Dropout(dropout)(x)

    # second conv block
    x = Conv2D(filter_size,
               num_filters[2],
               padding=padding,
               kernel_initializer=kernel_initializer)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    if isinstance(dropout, float):
        x = Dropout(dropout)(x)

    return x
