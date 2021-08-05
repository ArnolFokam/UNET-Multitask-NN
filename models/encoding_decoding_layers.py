from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import l2

from models.bottleneck_layers import bottleneck_layer
from models.resudial_blocks import resnet_block, res2net_block, dense_block


def block(x,
          filter_size,
          num_filters=3,
          activation='relu',
          padding='same',
          kernel_initializer='he_normal',
          dropout=None,
          variant='basic',
          _type='encoding'):
    assert variant in ['resnet', 'res2net', 'densenet', 'basic']
    assert _type in ['encoding', 'decoding']

    if variant == 'resnet':
        x = resnet_block(x,
                         filter_size=filter_size,
                         num_filters=num_filters,
                         activation=activation,
                         padding=padding,
                         kernel_initializer=kernel_initializer,
                         )
    elif variant == 'res2net':
        x = res2net_block(x,
                          filter_size=filter_size,
                          num_filters=num_filters,
                          activation=activation,
                          padding=padding,
                          kernel_initializer=kernel_initializer,
                          )
    elif variant == 'dense':
        x = dense_block(x,
                        num_layers=5,
                        filter_size=filter_size,
                        num_filters=num_filters,
                        activation=activation,
                        padding=padding,
                        dropout=dropout,
                        kernel_initializer=kernel_initializer,
                        )
    else:
        x = bottleneck_layer(x,
                             filter_size=filter_size,
                             num_filters=num_filters,
                             activation=activation,
                             padding=padding,
                             kernel_initializer=kernel_initializer,
                             )

    if _type == 'encoding':
        return x, MaxPooling2D(pool_size=(2, 2))(x)
    else:
        return x


def encoding_layer(x,
                   filter_size,
                   num_filters,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal',
                   variant='basic',
                   dropout=None):
    return block(x,
                 filter_size=filter_size,
                 num_filters=num_filters,
                 activation=activation,
                 padding=padding,
                 variant=variant,
                 dropout=dropout,
                 kernel_initializer=kernel_initializer,
                 )


def decoding_layer(x,
                   y,
                   filter_size,
                   num_filters,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal',
                   variant='basic',
                   dropout=None):
    x = UpSampling2D(size=(2, 2))(x)
    x = concatenate([x, y], axis=3)

    return block(x,
                 filter_size=filter_size,
                 num_filters=num_filters,
                 activation=activation,
                 padding=padding,
                 variant=variant,
                 kernel_initializer=kernel_initializer,
                 _type='decoding',
                 dropout=dropout)
