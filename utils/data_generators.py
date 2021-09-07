from abc import ABC
import numpy as np
import tensorflow as tf


def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow = '=' * int(percent/100 * barLength - 1) + '>'
    spaces = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')


class DicomDataGenerator(tf.keras.utils.Sequence, ABC):
    def __init__(self,
                 df,
                 img_path_col_name,
                 mask_path_col_name,
                 features_cols,
                 batch_size,
                 target_size=(512, 512, 1),
                 shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.target_size = target_size
        self.features_cols = features_cols
        self.img_path_col_name = img_path_col_name
        self.mask_path_col_name = mask_path_col_name
        self.shuffle = shuffle
        self.i = 0
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__get_data(batches)

    def __len__(self):
        return self.n // self.batch_size

    def __get_dicom_scan(self, path):
        # the scan are saved as numpy files
        image_arr = np.load(path).astype(np.float32)
        image_arr = tf.expand_dims(image_arr, axis=-1)

        img = tf.image.resize(image_arr, (self.target_size[0], self.target_size[1]), method='bicubic')

        # progress bar in dicom loading
        self.i += 1
        progressBar(self.i / 2, self.n, 100)
        return img

    def __get_data(self, batches):
        # get path of image data (change this function)
        img_path_batch = batches[self.img_path_col_name].values
        mask_path_batch = batches[self.mask_path_col_name].values
        # features = batches[self.features_cols].values

        X_batch = tf.convert_to_tensor([self.__get_dicom_scan(p) for p in img_path_batch])
        Y_mask_batch = tf.convert_to_tensor([self.__get_dicom_scan(p) for p in mask_path_batch])

        # normalize so it starts at 0
        Y_features_batch = tf.convert_to_tensor(batches[self.features_cols].values - 1, dtype=tf.int32)
        Y_malignancy_batch = tf.convert_to_tensor(batches['malignancy'].values - 1, dtype=tf.int32)
        return X_batch, Y_mask_batch, Y_features_batch, Y_malignancy_batch
