from collections import namedtuple

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

from models.encoding_decoding_layers import encoding_layer, decoding_layer

EncoderOutput = namedtuple('EncoderOutput', 'conv pool')

filters = [
    {
        'filter_size': 8,
        'dropout': None
    },
    {
        'filter_size': 16,
        'dropout': None
    },
    {
        'filter_size': 32,
        'dropout': 0.3
    },
    {
        'filter_size': 64,
        'dropout': None
    }]
kernel = 3


def PNSAMP_2D(
        num_attributes,
        pretrained_weights=None,
        input_size=(512, 512, 1),
        variant='basic'):
    inputs = Input(input_size)

    # ENCODER NETWORK PART
    encoders_output = list()
    for i in range(len(filters)):
        if i == 0:
            encoders_output.append(
                EncoderOutput(*encoding_layer(inputs,
                                              filters[i]['filter_size'],
                                              num_filters=kernel,
                                              dropout=filters[len(filters) - (i+1)]['dropout'],
                                              variant=variant)
                              ))
        else:
            encoders_output.append(
                EncoderOutput(*encoding_layer(
                    encoders_output[-1].pool,
                    filters[i]['filter_size'],
                    num_filters=kernel,
                    dropout=filters[len(filters) - (i+1)]['dropout'],
                    variant=variant))
            )

    # BETWEEN ENCODER AND DECODER
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        encoders_output[-1][1])
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # DECODER NETWORK PART
    decoders_output = None
    for i in range(len(filters)):
        if i == 0:
            decoders_output = decoding_layer(conv5,
                                             encoders_output[len(filters) - (i + 1)].conv,
                                             filters[len(filters) - (i + 1)]['filter_size'],
                                             num_filters=kernel,
                                             dropout=filters[len(filters) - (i+1)]['dropout'],
                                             variant=variant)
        else:
            decoders_output = decoding_layer(decoders_output,
                                             encoders_output[len(filters) - (i + 1)].conv,
                                             filters[len(filters) - (i + 1)]['filter_size'],
                                             num_filters=kernel,
                                             dropout=filters[len(filters) - (i+1)]['dropout'],
                                             variant=variant)

    if decoders_output is None:
        print('Please insert filters')
        return

    # SEGMENTATION OUTPUT
    conv10 = Conv2D(1, 1, activation='sigmoid', name="segmentation")(decoders_output)

    # SEGMENTATION OUTPUT FEATURE VECTOR
    pool10 = MaxPooling2D(pool_size=(2, 2))(conv10)
    dense10 = Dense(64, activation='sigmoid')(Flatten()(pool10))
    dense11 = Dense(8, activation='sigmoid')(dense10)

    # ENCODER OUTPUT FEATURE VECTOR
    dense5 = Dense(8, activation='sigmoid')(Flatten()(drop5))

    # MULTI-REGRESSION (fusion inspiration https://www.cs.unc.edu/~eunbyung/papers/wacv2016_combining.pdf)
    merge10 = Multiply()([dense5, dense11])
    dense12 = Dense(num_attributes, activation='sigmoid', name="multi_classification")(merge10)

    model = Model(inputs=[inputs], outputs=[conv10, dense12])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
