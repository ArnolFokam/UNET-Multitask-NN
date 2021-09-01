import os
import warnings
from configparser import ConfigParser

from preprocessing.data_preprocessor import DatasetPreprocessor
from preprocessing.utils import is_dir_path

warnings.filterwarnings(action='ignore')


def prepare_dataset():
    # I found out that simply using os.listdir() includes the gitignore file
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    test = DatasetPreprocessor(LIDC_IDRI_list[:100], DICOM_DIR, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK, META_DIR,
                               mask_threshold,
                               padding, confidence_level)
    test.prepare_dataset()


if __name__ == '__main__':
    # Read the configuration file generated from config_file_create.py
    parser = ConfigParser()
    parser.read('project.conf')

    # Get Directory setting
    DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
    MASK_DIR = is_dir_path(parser.get('prepare_dataset', 'MASK_PATH'))
    IMAGE_DIR = is_dir_path(parser.get('prepare_dataset', 'IMAGE_PATH'))
    CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
    CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
    META_DIR = is_dir_path(parser.get('prepare_dataset', 'META_PATH'))

    # Hyper Parameter setting for prepare dataset function
    mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')

    # Hyper Parameter setting for pylidc
    confidence_level = parser.getfloat('pylidc', 'confidence_level')
    padding = parser.getint('pylidc', 'padding_size')

    prepare_dataset()
