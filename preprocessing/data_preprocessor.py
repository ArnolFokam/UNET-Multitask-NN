import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import pydicom
import pylidc as pl
from tqdm import tqdm
from statistics import median_high

from pylidc.utils import consensus

from preprocessing.utils import segment_lung


def hu_normalize(im, slope, intercept):
    """normalize the image to Houndsfield Unit
    """
    im = im * slope + intercept
    im[im > 400] = 400
    im[im < -1000] = -1000

    im = (255 - 0) / (400 - (-1000)) * (im - 400) + 255

    return im.astype(np.uint8)


def get_any_file(path):
    files = glob.glob(path + "/*.dcm")
    if len(files) < 1:
        return None
    return pydicom.read_file(files[0])


class DatasetPreprocessor:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK, META_DIR,
                 mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.meta = pd.DataFrame(index=[],
                                 columns=[
                                     'patient_id',
                                     'nodule_no',
                                     'slice_no',
                                     'original_image',
                                     'mask_image',
                                     'subtlety',
                                     'internalStructure',
                                     'calcification',
                                     'sphericity',
                                     'margin',
                                     'lobulation',
                                     'spiculation',
                                     'texture',
                                     'malignancy',
                                     'is_cancer',
                                     'is_clean'])

    def calculate_malignancy(self, nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the
        # annotated cancer, True or False label for cancer if median high is above 3, we return a label True for
        # cancer if it is below 3, we return a label False for non-cancer if it is 3, we return ambiguous
        list_of_malignancy = []
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self, meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,
                        index=[
                            'patient_id',
                            'nodule_no',
                            'slice_no',
                            'original_image',
                            'mask_image',
                            'subtlety',
                            'internalStructure',
                            'calcification',
                            'sphericity',
                            'margin',
                            'lobulation',
                            'spiculation',
                            'texture',
                            'malignancy',
                            'is_cancer',
                            'is_clean'])
        self.meta = self.meta.append(tmp, ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            """dcm = get_any_file(scan.get_path_to_dicom_files())
            intercept = dcm.RescaleIntercept
            slope = dcm.RescaleSlope"""
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, vol.shape,
                                                                                          len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                    # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    lung_np_array = vol[cbbox]
                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)

                    # compute the average value for each attributes annotation
                    # from the different radiologists
                    subtlety = median_high([ann.subtlety for ann in nodule])
                    internalStructure = median_high([ann.internalStructure for ann in nodule])
                    calcification = median_high([ann.calcification for ann in nodule])
                    sphericity = median_high([ann.sphericity for ann in nodule])
                    margin = median_high([ann.margin for ann in nodule])
                    lobulation = median_high([ann.lobulation for ann in nodule])
                    spiculation = median_high([ann.spiculation for ann in nodule])
                    texture = median_high([ann.texture for ann in nodule])

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                            continue

                        # Segment Lung part only
                        lung_segmented_np_array = segment_lung(lung_np_array[:, :, nodule_slice])
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in
                        # pytorch training # Not sure
                        lung_segmented_np_array[lung_segmented_np_array == -0] = 0
                        # This iterates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                        meta_list = [
                            pid[-4:],
                            nodule_idx,
                            prefix[nodule_slice],
                            nodule_name,
                            mask_name,
                            subtlety,
                            internalStructure,
                            calcification,
                            sphericity,
                            margin,
                            lobulation,
                            spiculation,
                            texture,
                            malignancy,
                            cancer_label,
                            False]

                        self.save_meta(meta_list)
                        # np.save(patient_image_dir / nodule_name, lung_segmented_np_array)
                        # np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])
            else:
                print("Clean Dataset", pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                # There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this
                # for validation
                for slice in range(vol.shape[2]):
                    if slice > 50:
                        break
                    lung_segmented_np_array = segment_lung(vol[:, :, slice])
                    lung_segmented_np_array[lung_segmented_np_array == -0] = 0
                    lung_mask = np.zeros_like(lung_segmented_np_array)

                    # CN= CleanNodule, CM = CleanMask
                    nodule_name = "{}_CN001_slice{}".format(pid[-4:], prefix[slice])
                    mask_name = "{}_CM001_slice{}".format(pid[-4:], prefix[slice])
                    meta_list = [pid[-4:],
                                 slice,
                                 prefix[slice],
                                 nodule_name,
                                 mask_name,
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 'N/A',
                                 0,
                                 False,
                                 True]
                    self.save_meta(meta_list)
                    # np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
                    # np.save(patient_clean_dir_mask / mask_name, lung_mask)

        print("Saved Meta data")
        self.meta.to_csv(self.meta_path + 'meta_info.csv', index=False)