import argparse
import os
import shutil
import sys
import tempfile
import time
import zipfile
import requests
import pandas as pd
import utils
from plotting_utils import plot_and_save_ims
from deploy_constants import *


def zip_files(path):
    """
    Create a temporary zip file with all the files in a given path
    :param path: str. Path to the file/folder
    :return: str. Path to the temporary file
    """
    tmp_file = tempfile.mktemp(suffix="_upload.zip")
    # Use max compression
    zip_fp = zipfile.ZipFile(tmp_file, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
    if os.path.isfile(path):
        zip_fp.write(path)
    else:
        for root, dirs, fns in os.walk(path):
            for fn in fns:
                filepath = os.path.abspath(os.path.join(root, fn))
                zip_fp.write(filepath, os.path.join(os.path.relpath(root, path), fn))


    zip_fp.close()
    # Check if max file size is exceeded
    file_size = os.stat(tmp_file).st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        os.remove(tmp_file)
        raise Exception(f"Max file size exceeded, the file cannot be uploaded ({file_size / 2**30} GB)." \
                        f" Please limit the uploaded file size to {MAX_FILE_SIZE_BYTES / 2**30} GB")
    return tmp_file


def send_file(zip_file, access_key):
    """
    Send a zip file to the remote web server for processing
    :param zip_file: str. Path to the zip file to send
    :param access_key: str. Client ID
    :return: tuple with SessionID assigned by the server and expected results url
    """
    r = requests.post(SERVER_IP + "/new", json={'sender': access_key}, timeout=60)
    if r.status_code != 200:
        raise Exception("The server could not be reached or the received response is incorrect: {}".format(r.text))
    if r.text.startswith("ERROR"):
        raise Exception("The server returned the following error: {}".format(r.text))
    session_id = r.text
    headers = {
        'SessionId': session_id,
        'AccessKey': access_key,
    }
    fname = os.path.basename(zip_file)
    files = {'zip_file': (fname, open(zip_file, 'rb'), 'application/zip',
                          {'session_id': session_id, 'access_key': access_key})}

    r = requests.post(SERVER_IP + "/upload", files=files, headers=headers)

    if r.status_code != 200:
        raise Exception("Error uploading files: {}".format(r.text))
    if r.text.startswith("ERROR"):
        raise Exception("The server returned the following error when uploading the files: {}".format(r.text))
    expected_results_url = r.text
    return session_id, expected_results_url


def download_remote_results(expected_url, local_file_path, waiting_time_seconds=30, max_attempts=100):
    """
    Check if a remote file exists and download it.
    If it's found, download it to a local output folder.
    If file is not found after some time, return None
    :param expected_url: str. Remote url
    :param local_file_path: str. Local results file path
    :return: str. Path to the local file once it has been downloaded, or None
    """
    i = 0
    while i < max_attempts:
        req = requests.get(expected_url)
        if req.status_code == 200:
            # File found
            with open(local_file_path, 'wb') as f:
                f.write(req.content)
            return local_file_path

        print("Results are being generated. Please wait...")
        max_attempts += 1
        time.sleep(waiting_time_seconds)
    # Timeout error
    return None


def unhash_results(results_zip_file, input_, output_dir):
    """
    Unhash filepaths in results dataframes
    :param results_zip_file: str. Local path to the results zip file
    :param input_: str. Input folder or file
    :param output_dir: str. Output folder
    :return: 2-tuple dataframe: study_df, dicom_df
    """
    # Recreate hash map so we can convert to true paths
    hash_map = dict()
    if os.path.isfile(input_):
        root = '/'.join(input_.split('/')[0:-1])
        hash_map[utils.create_hash(root)] = root
        hash_map[utils.create_hash(input_)] = input_
    else:
        for root, dirs, files in os.walk(input_):
            # Assume we are in a study folder when it only contains files
            if len(dirs) == 0 and len(files) > 0:
                hash_map[utils.create_hash(root)] = root
                for file in files:
                    hash_map[utils.create_hash(os.path.join(root, file))] = os.path.join(root, file)

    # Unzip results_file
    session_id = results_zip_file.split('/')[-1].split('.zip')[0]
    unzip_destination = os.path.join(output_dir, session_id, "csv")
    with zipfile.ZipFile(results_zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_destination)

    # Replace hash paths in study_df
    study_df_path = os.path.join(unzip_destination, session_id + '_study.csv')
    study_df = pd.read_csv(study_df_path)
    # Ensure the hashes are in a string format
    study_df['StudyInstanceUID'] = study_df['StudyInstanceUID'].astype(str)
    study_df['study_path'] = ''
    for idx, row in study_df.iterrows():
        study_df.loc[idx, 'study_path'] = hash_map[row['StudyInstanceUID']]

    # Replace hash paths in dicom_df
    dicom_df_path = os.path.join(unzip_destination, session_id + '_dicom.csv')
    dicom_df = pd.read_csv(dicom_df_path)
    # Ensure the hashes are in a string format
    dicom_df['StudyInstanceUID'] = dicom_df['StudyInstanceUID'].astype(str)
    dicom_df['SOPInstanceUID'] = dicom_df['SOPInstanceUID'].astype(str)

    dicom_df['study_path'] = ''
    dicom_df['file_path'] = ''
    for idx, row in dicom_df.iterrows():
        dicom_df.loc[idx, 'study_path'] = hash_map[row['StudyInstanceUID']]
        dicom_df.loc[idx, 'file_path'] = hash_map[row['SOPInstanceUID']]

    study_df = study_df[['study_path', 'score']]
    study_df.to_csv(study_df_path, index=False)
    dicom_df = dicom_df[['file_path', 'x1', 'y1', 'x2', 'y2', 'slice', 'score']]
    dicom_df.to_csv(dicom_df_path, index=False)
    return study_df, dicom_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation of AI algorithms for cancer detection in mammography.\n"
                                                 "Example of use: python deploy_evaluation.py "
                                                 "--input /my/local/dicom_data "
                                                 "--output /my/local/output/folder "
                                                 "--access_key XXX")
    parser.add_argument('--input', type=str, required=True,
                        help='Path can be a DICOM file, a study directory containing DICOM files, or directory of study directories')
    parser.add_argument('--output', type=str, required=True, help='Output directory to store results')
    parser.add_argument('--access_key', type=str, required=True, help='Access key provided by the authors')
    parser.add_argument('--preprocess_dir', type=str, help='Directory to store pre-processed files (optional)')
    parser.add_argument('--results_url', type=str,
                        help='Results url for a previous request (used to display the results asynchronously)')
    parser.add_argument('--plot_images', type=str, choices=('y', 'n'), help="Generate images/bounding boxes (y/n)")

    args = parser.parse_args()

    # If input is single file, treat parent folder like study folder
    input_ = os.path.realpath(args.input)

    if args.results_url is None:
        # Process inputs
        if args.preprocess_dir is None:
            preprocess_dir = tempfile.mkdtemp(prefix="preprocessed_")
            keep_preprocessed_dir = False
        else:
            preprocess_dir = args.preprocess_dir
            if os.path.exists(preprocess_dir):
                assert os.path.isdir(preprocess_dir) and len(os.listdir(preprocess_dir)) > 0, \
                        "Please use a new or empty folder as 'preprocess_dir' parameter"
            else:
                os.makedirs(preprocess_dir)
            keep_preprocessed_dir = True

        # Populate preprocessed_dir
        print("Reading files...")

        # If input is single file, treat parent folder like study folder
        num_images = 0
        if os.path.isfile(input_):
            root = '/'.join(input_.split('/')[0:-1])
            metadata = utils.read_study(root, preprocess_dir, input_)
            num_images += len(metadata)
            assert num_images <= 1, f"ERROR: The input of {input_} is one file but more than one image would be uploaded."
        elif os.path.isdir(input_):
            num_studies = 0
            for root, dirs, files in os.walk(input_):
                # Assume we are in a study folder when it only contains files
                if len(dirs) == 0 and len(files) > 0:
                    num_studies += 1
                    num_images_preview = len(files)

                    # Enforce Evaluation Limit
                    if num_studies > MAX_STUDIES or (num_images + num_images_preview) > MAX_IMAGES:
                        print(f"WARNING: Enforcing Evaluation Limit of {MAX_STUDIES} studies and {MAX_IMAGES} images.")
                        break

                    # Process folder
                    metadata = utils.read_study(root, preprocess_dir)
                    # Update the number of images with the real number of preprocessed images
                    num_images += len(metadata)

        else:
            raise Exception('Path {} does not exist.'.format(input_))

        if num_images == 0:
            print(f"No valid files were found in input '{input_}'")
            sys.exit(0)

        print("Preparing files for sending...")
        zip_file_path = zip_files(preprocess_dir)

        print("Files are ready to send. Please confirm the following statement to proceed:")
        print(f"I agree with the Terms of Service {TERMS_LINK} set by DeepHealth and certify that the images transferred do not include protected health information.")
        n_attempts = 0
        max_attempts = 5
        while n_attempts < max_attempts:
            answer = str(input('Confirm (y/n)?')).lower().strip()
            if answer == 'y':
                break
            n_attempts += 1
            if answer == 'n' or n_attempts >= max_attempts:
                print('You must agree to the Terms of Service to proceed.')
                quit()

        try:
            print("Sending files to the server (temp files in {})...".format(zip_file_path))
            session_id, expected_results_remote = send_file(zip_file_path, args.access_key)
        finally:
            print("Cleaning temp files...")
            if not keep_preprocessed_dir:
                shutil.rmtree(preprocess_dir)
            os.remove(zip_file_path)
        print("The results are being generated.\n"
              "You can stop this process now pressing Ctrl+C or wait for the results to be generated\n"
              "If you decide to stop the process, you can obtain and display the results later using this url as 'results_url' parameter:\n"
              "{}".format(expected_results_remote))
    else:
        # The input files were already sent and just results should be displayed
        expected_results_remote = args.results_url
        session_id = os.path.basename(expected_results_remote).replace(".zip", "")

    results_local_file_path = os.path.join(args.output, "{}.zip".format(session_id))
    print("Waiting for results to be ready in {}...".format(expected_results_remote))
    results_file = download_remote_results(expected_results_remote, results_local_file_path)
    if results_file:
        print(f"Results downloaded! See {results_local_file_path}")
        session_id = os.path.basename(results_local_file_path).replace(".zip", "")
        results_folder = os.path.join(args.output, session_id)
        # Results available
        # Generate dataframes
        study_df, dicom_df = unhash_results(results_file, input_, args.output)
        # Generate images/bounding boxes (optional)
        if len(dicom_df) > 0:
            if args.plot_images == 'y':
                # If the parameter was set then images will be generated
                plot_images = True
            elif args.plot_images == 'n':
                plot_images = False
            else:
                answer = ''
                while answer not in ('y', 'n'):
                    answer = str(input("Do you want to generate images for the results? (y/n)")).lower().strip()
                plot_images = answer == 'y'
            if plot_images:
                plot_and_save_ims(dicom_df, os.path.join(args.output, session_id))
                print("Images generated!")
        print("All the results files have been saved to the folder {}".format(results_folder))
        # Remove temp file
        os.remove(results_file)
    else:
        print("Results could not be found. Please try again later or contact the administrator")
