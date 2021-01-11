import os
import logging
import hashlib

import pydicom
import numpy as np
import pandas as pd

def create_hash(string):
    return hashlib.sha256(string.encode('ASCII')).hexdigest()[0:8]

def verbose_position_to_code(string):
    # dictionary based on http://dicom.nema.org/dicom/2013/output/chtml/part16/sect_CID_4014.html
    CODE_TO_ACRONYM = {'medio-lateral': 'ML',
                       'medio-lateral oblique': 'MLO',
                       'latero-medial': 'LM',
                       'latero-medial oblique': 'LMO',
                       'cranio-caudal': 'CC',
                       'caudo-cranial (from below)': 'FB',
                       'superolateral to inferomedial oblique': 'SIO',
                       'inferomedial to superolateral oblique': 'ISO',
                       'exaggerated cranio-caudal': 'XCC', # not used any more
                       'cranio-caudal exaggerated laterally': 'XCCL',
                       'cranio-caudal exaggerated medially': 'XCCM',
                       'tissue specimen from breast': 'SPECIMEN'
                      }
    if string in CODE_TO_ACRONYM:
        return CODE_TO_ACRONYM[string]
    elif string.lower().strip() == 'mediolateral oblique':
        return 'MLO'
    elif string.lower().strip() == 'craniocaudal':
        return 'CC'
    else:
        raise Exception('Could not find view code match for ' + string)

def nested_index(data, index_array_in):
    # Copy so we don't change the original input
    index_array = index_array_in.copy()
    # Takes in array of indicies, ex. [["0054", "0220"], 0, ["0008", "0104"]]
    # Recursive function to access nested values
    if len(index_array) == 0:
        return data
    else:
        try:
            data = data[index_array.pop(0)]
        except:
            raise ValueError('Value not found')
        return nested_index(data, index_array)

def read_dicom(dicom_path, preprocess_dir, study_path_hash):
    def map_manufacturer(manufacturer):
        if 'hologic' in manufacturer.lower() or 'lorad' in manufacturer.lower():
            return 'hologic'
        elif 'gemedicalsystems' in manufacturer.replace(' ', '').lower():
            return 'ge'
        else:
            return manufacturer
    # Read in dicom
    ds = pydicom.dcmread(dicom_path)
    is_dbt = ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.13.1.3'

    # Preprocessing
    # Ensure the SOPClassUID is valid
    allowed_sop_uids = ('1.2.840.10008.5.1.4.1.1.1.2', '1.2.840.10008.5.1.4.1.1.13.1.3')
    assert ds.SOPClassUID in allowed_sop_uids, \
        f"SOPClassUID incorrect ({ds.SOPClassUID}). Allowed values: {allowed_sop_uids}"
    # Ensure that there are no burned in annotations in the images, which could contain PHI
    assert 'BurnedInAnnotation' in ds and ds.BurnedInAnnotation.upper() == 'NO', \
        'BurnedInAnnotation not found or != "NO"'

    # Hash path in case the path contains PHI
    dcm_path_hash = create_hash(dicom_path)

    # Parse metadata fields and store in dictionary
    metadata = dict()
    dicom_attributes = ['SOPClassUID',
                        'ViewPosition',
                        'ImageLaterality',
                        'Rows',
                        'Columns',
                        'Manufacturer',
                        'ManufacturerModelName',
                        'NumberOfFrames',
                        'PatientOrientation',
                        'BitsAllocated',
                        'WindowWidth',
                        'WindowCenter',
                        'WindowCenterWidthExplanation',
                        'HighBit']

    dbt_fields = {'ViewPosition': [["0054", "0220"], 0, ["0008", "0104"]],
                 'ImageLaterality': [["0019", "108a"], 0, ["0019", "1087"]],
                 'WindowWidth': [["5200", "9229"], 0, ["0028", "9132"], 0, ["0028", "1051"]],
                 'WindowCenter': [["5200", "9229"], 0, ["0028", "9132"], 0, ["0028", "1050"]]}

    for field in dicom_attributes:
        if field in dbt_fields and is_dbt:
            try:
                field_val = nested_index(ds, dbt_fields[field]).value
                if field == 'ViewPosition':
                    field_val = verbose_position_to_code(field_val)
                if field == 'ImageLaterality':
                    field_val = field_val[0]
                if field in ['WindowWidth', 'WindowCenter']:
                    field_val = int(field_val)
            except ValueError:
                field_val = None
        else:
            field_val = getattr(ds, field, None)
        if field_val is None:
            field_val = ''
        metadata[field] = field_val

    metadata['dcm_path'] = dcm_path_hash
    metadata['np_paths'] = os.path.join(study_path_hash, dcm_path_hash)
    metadata['SOPInstanceUID'] = dcm_path_hash
    metadata['StudyInstanceUID'] = study_path_hash

    # Convert pixel_array into dictionary of frames
    pxl_array = ds.pixel_array
    pxl_dict = dict()
    num_frames = metadata['NumberOfFrames']
    # If the manufacturer is not Hologic or GE, apply pydicom windowing. 
    manufacturer = map_manufacturer(str(metadata['Manufacturer']))
    if manufacturer.lower() not in ['ge', 'hologic']:
        pxl_array = pydicom.pixel_data_handlers.apply_voi_lut(pxl_array, ds)
    if num_frames is '':
        pxl_dict[0] = pxl_array
    else:
        assert pxl_array.shape[0] == num_frames, \
            f"The NumberOfFrames dicom metadata field ({num_frames}) and the pixel data shape ({pxl_array.shape})" \
            " are inconsistent"
        for i in range(int(num_frames)):
            pxl_dict[i] = pxl_array[i]

    # Save dictionary of frames to numpys
    dicom_preprocess_dest = os.path.join(preprocess_dir, study_path_hash, dcm_path_hash)
    os.makedirs(dicom_preprocess_dest, exist_ok=True)
    for i, frame in pxl_dict.items():
        path_ = os.path.join(dicom_preprocess_dest, 'frame_{}.npy'.format(i))
        np.save(path_, frame)

    return metadata

def read_study(study_path, preprocess_dir, only_include=None):
    # Hash study_path in case it has PHI
    study_path_hash = create_hash(study_path)
    study_metadata = []

    # Iterate through and read each dicom
    for dicom in os.listdir(study_path):
        dicom_path = os.path.join(study_path, dicom)

        if only_include is not None and dicom_path != only_include:
            continue
        try:
            dicom_metadata = read_dicom(dicom_path, preprocess_dir, study_path_hash)
        except Exception as ex:
            print("File {} could not be processed correctly as a DICOM file ({}) (skipped)".format(dicom_path, ex))
            continue
        study_metadata.append(dicom_metadata)

    # Convert study_metadata into a Pandas DataFrame
    study_metadata = pd.DataFrame.from_dict(study_metadata, orient='columns')

    # Save study_metadata in preprocessed_dir
    study_preprocess_dest = os.path.join(preprocess_dir, study_path_hash)
    os.makedirs(study_preprocess_dest, exist_ok=True)
    study_metadata.to_pickle(os.path.join(study_preprocess_dest, 'study_metadata.pkl'))
    return study_metadata
