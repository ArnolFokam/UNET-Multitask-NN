import shutil

import numpy as np
import os
import cv2
import requests
import concurrent.futures

GET_SINGLE_IMAGE = "getSingleImage"
GET_PATIENT_STUDY = "getPatientStudy"
GET_SERIES = "getSeries"
GET_PATIENT_BY_MODALITY = "PatientsByModality"
GET_ALL_SOP_INSTANCES_UID = 'getSOPInstanceUIDs'

PATIENT_ID = 'PatientID'
STUDY_INSTANCE_UID = 'StudyInstanceUID'
SERIES_INSTANCE_UID = 'SeriesInstanceUID'
SOP_INSTANCE_UID = 'SOPInstanceUID'

baseUrl = 'https://services.cancerimagingarchive.net/services/v4/TCIA/query/'

defaultParameters = {
    'Collection': 'LIDC-IDRI',
    'Modality': 'CT'
}

np.random.seed(42)
session = requests.session()


def array_to_rgb(x):
    x = np.load(x)

    # Converted the datatype to np.uint8
    array = x.astype(np.uint8) * 255

    # stack the channels in the new image
    rgb = np.dstack([array, array, array])
    return rgb


def download_image(url, img_path, params):
    img_bytes = session.get(url, params=params).content
    with open(img_path, 'wb') as img_file:
        print(img_path)
        img_file.write(img_bytes)
    return img_bytes


def read_image(path, IMAGE_SIZE):
    x = cv2.imread(path)
    x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    x = x / 255.0
    return x


def execute(url, queryParameters=None):
    if queryParameters is None:
        queryParameters = defaultParameters

    response = session.get(baseUrl + url,
                           params=queryParameters)
    return response


def fetch_patients(num_of_patients=None):
    resp = execute(GET_PATIENT_BY_MODALITY)
    if num_of_patients is None:
        return resp.json()
    return resp.json()[:num_of_patients]


def fetch_patient_study(patient_id):
    parameters = defaultParameters
    parameters['PatientID'] = patient_id
    print('Retrieving the studies for patient ', patient_id)
    response = execute(GET_PATIENT_STUDY)
    return response.json()


def fetch_study_series(patient_id, study_id):
    parameters = defaultParameters
    parameters['PatientID'] = patient_id
    parameters['StudyInstanceUID'] = study_id
    print('Retrieving the series of study {} for patient {}'.format(study_id, patient_id))
    response = execute(GET_SERIES)
    return response.json()


def fetch_dicom_images_by_series(series_id, path):
    parameters = defaultParameters
    parameters[SERIES_INSTANCE_UID] = series_id

    url = baseUrl + GET_ALL_SOP_INSTANCES_UID

    # get all dicom file names with their SOPInstanceUID attached
    dicom_files = session.get(url, params=parameters).json()

    url = baseUrl + GET_SINGLE_IMAGE

    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        print("Creation of the directory %s failed" % path)
        print("Could not save images of series ", series_id)
    else:
        print("Successfully created the directory %s " % path)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dicom in dicom_files:
                parameters[SOP_INSTANCE_UID] = dicom[SOP_INSTANCE_UID]
                executor.submit(download_image,
                                url,
                                os.path.join(
                                    path,
                                    dicom[SOP_INSTANCE_UID] + '.dcm'),
                                parameters)


# TODO: write argument support
if __name__ == '__main__':

    # 1. Fetch Patients
    patients = fetch_patients(500)

    # 2. Fetch Studies
    studies = {}
    for patient in patients:
        studies[patient[PATIENT_ID]] = fetch_patient_study(patient[PATIENT_ID])

    # 3. Fetch Studies Series
    study_series = dict()
    for key in studies:
        for study in studies[key]:
            study_series[study[STUDY_INSTANCE_UID]] = fetch_study_series(study[PATIENT_ID], study[STUDY_INSTANCE_UID])

    # 4. Fetch Studies
    scans_folder = './data/scans'
    shutil.rmtree(scans_folder, ignore_errors=True)
    for key in study_series:
        for series in study_series[key]:
            fetch_dicom_images_by_series(
                series[SERIES_INSTANCE_UID],
                os.path.join(
                    scans_folder,
                    series[PATIENT_ID],
                    series[STUDY_INSTANCE_UID],
                    series[SERIES_INSTANCE_UID]))
