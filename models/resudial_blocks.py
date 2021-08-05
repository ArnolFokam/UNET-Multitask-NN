from tensorflow.keras.layers import *

from models.bottleneck_layers import bottleneck_dense_layer, bottleneck_resnet_layer, bottleneck_layer


def resnet_block(x,
                 filter_size,
                 num_filters=3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal',):
    conv = bottleneck_resnet_layer(x,
                                   filter_size,
                                   num_filters,
                                   activation=activation,
                                   padding=padding,
                                   kernel_initializer=kernel_initializer,
            )
    x = Conv2D(filter_size,
               1,
               padding=padding,
               kernel_initializer=kernel_initializer,
               )(x)

    # add residual connection
    x = Add()([x, conv])
    x = Activation(activation)(x)

    return x


def res2net_block(x,
                  filter_size,
                  num_filters=3,
                  activation='relu',
                  padding='same',
                  kernel_initializer='he_normal',
                  scale_dim=4,):
    assert isinstance(num_filters, int)

    # 1x1 conv block
    conv = Conv2D(filter_size,
                  1,
                  padding=padding,
                  kernel_initializer=kernel_initializer,
                  )(x)
    conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)

    assert filter_size % scale_dim == 0

    channels_per_group = filter_size // scale_dim
    subset_out = []

    # multi scale system
    for i in range(scale_dim):
        slice_conv = Lambda(lambda y: y[..., i * channels_per_group:(i + 1) * channels_per_group])(conv)

        if i > 1:
            slice_conv = Add()([slice_conv, subset_out[-1]])
        if i > 0:
            slice_conv = bottleneck_layer(slice_conv,
                                          filter_size=channels_per_group,
                                          num_filters=num_filters,
                                          activation=activation,
                                          padding=padding,
                                          kernel_initializer=kernel_initializer
                                          )

        subset_out.append(slice_conv)
    conv = Concatenate()(subset_out)

    # 1x1 conv block in multi-scale
    conv = Conv2D(filter_size,
                  1,
                  padding=padding,
                  kernel_initializer=kernel_initializer
                  )(conv)
    conv = BatchNormalization()(conv)

    # residual connection
    conv_res = Conv2D(filter_size,
                      1,
                      padding=padding,
                      kernel_initializer=kernel_initializer
                      )(x)

    conv = Add()([conv, conv_res])
    conv = Activation(activation)(conv)

    return conv


def dense_block(input_x,
                num_filters,
                filter_size,
                num_layers=3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
                dropout=None):
    assert num_layers >= 3

    layers_concat = list()
    layers_concat.append(input_x)

    x = bottleneck_dense_layer(x=input_x,
                               num_filters=(1, num_filters),
                               filter_size=filter_size,
                               activation=activation,
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               dropout=dropout
                               )

    layers_concat.append(x)

    for i in range(num_layers - 1):
        x = concatenate(layers_concat)
        x = bottleneck_dense_layer(x=x,
                                   num_filters=(1, num_filters),
                                   filter_size=filter_size,
                                   activation=activation,
                                   padding=padding,
                                   kernel_initializer=kernel_initializer,
                                   dropout=dropout
                                   )
        layers_concat.append(x)

    x = concatenate(layers_concat)

    return x
