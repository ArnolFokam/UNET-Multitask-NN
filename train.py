import os
from configparser import ConfigParser

import tensorflow as tf
import pandas as pd
from models.pnsamp_2d import PNSAMP_2D
from sklearn.model_selection import train_test_split

from preprocessing.utils import is_dir_path
from utils.data_generators import DicomDataGenerator

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('project.conf')

# Get Directory setting
data_path = is_dir_path(parser.get('train', 'DATA_PATH'))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(data_path, 'meta/meta_info.csv'),
                     dtype={'patient_id': str,
                            'nodule_no': str,
                            'slice_no': str})
    # use only non-clean scans (scans that contains at least one nodule) for training
    df = df[df['is_clean'] == False]


    def get_paths(x):
        patient_img_path = os.path.join(data_path, 'image', 'LIDC-IDRI-' + x[0])
        patient_mask_path = os.path.join(data_path, 'mask', 'LIDC-IDRI-' + x[0])
        return [os.path.join(patient_img_path, x[1] + '.npy'), os.path.join(patient_mask_path, x[2] + '.npy')]


    temp = df[['patient_id',
               'original_image',
               'mask_image']].values

    paths = list(map(get_paths, temp))
    df_paths = pd.DataFrame(paths, columns=['img_path', 'mask_path'])

    df.reset_index(drop=True, inplace=True)
    df_paths.reset_index(drop=True, inplace=True)

    df = pd.concat([df, df_paths], axis=1, sort=False)

    df = df[[
        'subtlety',
        'internalStructure',
        'calcification',
        'sphericity',
        'margin',
        'lobulation',
        'spiculation',
        'texture',
        'malignancy',
        'img_path',
        'mask_path'
    ]]

    from sklearn import preprocessing

    scaler = preprocessing.MinMaxScaler()

    df[[
        'subtlety',
        'internalStructure',
        'calcification',
        'sphericity',
        'margin',
        'lobulation',
        'spiculation',
        'texture',
        'malignancy',
    ]] = scaler.fit_transform(df[[
        'subtlety',
        'internalStructure',
        'calcification',
        'sphericity',
        'margin',
        'lobulation',
        'spiculation',
        'texture',
        'malignancy',
    ]])

    batch_size = 5
    image_size = 128
    num_batches = int(len(df) / batch_size)
    datagen = DicomDataGenerator(df,
                                 img_path_col_name='img_path',
                                 mask_path_col_name='mask_path',
                                 features_cols=['subtlety',
                                                'internalStructure',
                                                'calcification',
                                                'sphericity',
                                                'margin',
                                                'lobulation',
                                                'spiculation',
                                                'texture'],
                                 batch_size=batch_size,
                                 target_size=(image_size, image_size, 1)
                                 )

    traingen, valigen = train_test_split(datagen, test_size=0.33)

    model = PNSAMP_2D(num_attributes=8, input_size=(image_size, image_size, 1), variant='res2net')
    model.summary()

    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()

    # Instantiate a loss function.
    loss_fn1 = tf.keras.losses.BinaryCrossentropy()
    loss_fn2 = tf.keras.losses.BinaryCrossentropy()

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(traingen):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape(persistent=True) as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                segmentation_logits, multi_regr_logits = model(x_batch_train,
                                                               training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value1 = loss_fn1(y_batch_train[0], segmentation_logits)
                loss_value2 = loss_fn2(y_batch_train[1], multi_regr_logits)
                print(loss_value1.numpy(), loss_value2.numpy())

            """MULTI-CLASSIFICATOIN"""
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient([loss_value1, loss_value2], model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            """if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))"""
