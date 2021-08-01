from abc import ABC
import numpy as np
import cv2
import tensorflow as tf


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
        self.n = len(self.df)

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        return self.n // self.batch_size

    def __get_dicom_scan(self, path):
        # the scan are saved as numpy files
        image_arr = np.load(path).astype(np.float32)
        image_arr = cv2.resize(image_arr, (self.target_size[0], self.target_size[1]), interpolation=cv2.INTER_AREA)
        image_arr / np.amax(image_arr)

        return tf.expand_dims(image_arr, axis=-1)

    def __get_data(self, batches):
        # get path of image data (change this function)
        img_path_batch = batches[self.img_path_col_name].values
        # mask_path_batch = batches[self.mask_path_col_name].values

        X_batch = np.asarray([self.__get_dicom_scan(p) for p in img_path_batch])
        y_batch = tuple([[self.__get_dicom_scan(p) for p in batches[self.mask_path_col_name].values],
                         batches[self.features_cols].values])

        """y_batch = [
            [self.__get_dicom_scan(p), features] for p, *features in
            batches[[self.mask_path_col_name] + self.features_cols].values]"""

        return X_batch, y_batch
