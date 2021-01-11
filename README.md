## Overview

This repo contains code to enable evaluation of the mammography AI algorithms described in Lotter W, Diab AR, Haslam B, Kim JG, Grisot G, Wu E, Wu K, Onieva Onieva J, Boyer Y, Boxerman JL, Wang M, Bandler M, Vijayaraghavan GR, and Sorensen AG. Robust breast cancer detection in mammography and digital breast tomosynthesis using an annotation-efficient deep learning approach. _Nature Medicine_ (2021).

Evaluation occurs via a number of steps:
1. User provides a local path to mammography DICOM data.
2. Pixel data and non-PHI metadata (e.g., view, laterality, etc.) are extracted locally and saved as a zip file.
3. The zip file is sent to an evaluation server where model inference occurs.
4. Results including study-level and file-level csv's are sent back to the user.
5. (Optional): Bounding boxes are plotted on the input images.

DeepHealth, the AI division of RadNet, is providing the evaluation server and accompanying code to support the scientific research community. For any issues or support, please contact [evalserversupportdeephealth@radnet.com](mailto:evalserversupportdeephealth@radnet.com).


## User Requirements and Restrictions
* Code is available for research purposes only. CLINICAL, DIAGNOSTIC, TREATMENT AND COMMERCIAL USES ARE PROHIBITED.
* Code must not be run on any images that contain protected health information (PHI) in the pixel data. More information on PHI can be found at https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
* In order to use the provided code to send data to the evaluation server, you must be granted an access key first.  Please contact [evalserversupportdeephealth@radnet.com](mailto:evalserversupportdeephealth@radnet.com) to retrieve your access key.
* Use of the provided scripts and evaluation server is subject to our Evaluation Terms of Service contained in the following link on the DeepHealth website: https://deep.health/evaluation-terms-of-service/. By using the provided access key, you are agreeing to be bound by the Evaluation Terms of Service. Upon running the scripts to send data to the evaluation server, you will additionally be asked to confirm your agreement to our Evaluation Terms of Service.
* Each user is limited to the evaluation of up to 100 studies or 500 images (whichever limit is reached first). This will help prevent the server from being overloaded and ensure fair distribution across users.  


## Use Instructions

The main script that facilitates evaluation and analysis is ```deploy_evaluation.py```. An example of how this script can be run is provided below:

```
python deploy_evaluation.py --input /my/local/dicom_data --output
/my/local/output/folder --access_key XXX
```

In more detail, this script takes the following command line arguments:
* ```--input```: Path to DICOM data. This path can be a single DICOM file, a study directory containing the DICOM files for a study, or a directory of study directories. (Required)
* ```--output```: Path to a directory to store the evaluation results. (Required)
* ```--access_key```: Your unique access key that has been provided to you, as described above. (Required)
* ```--preprocess_dir```: Path to a directory to store pre-processed files. Before sending data to the evaluation server, some preprocessing occurs, including the extraction of pixel arrays and certain metadata. This pre-processed data is saved locally before being compressed and sent to the evaluation server. If ```--preprocess_dir``` is provided, the pre-processed data will be stored in this directory, otherwise a temporary directory is created. (Optional)
* ```--results_url```: URL from a previous request that can be used to download the results. When ```deploy_evaluation.py``` is initially run and data is sent to the server, the script will then display the URL where the evaluation results will be available and then periodically check this URL for the results. If the results aren't available after a certain number of attempts, the script will then timeout. The ```--results_url``` parameter can then be used at a later time to retrieve the results once they are available. If the ```--results_url``` parameter is used, then the ```--input``` data parameter will only be used to post-process the results and the input data will not be sent to the evaluation server again. (Optional)
* ```--plot_images```: Whether or not to generate images with bounding boxes from the results ('y' or 'n'). If this parameter is set to 'y', images with bounding boxes will be generated from the returned results. If the parameter is not set, you will be asked if you want to generate the bounding box images after the results have been downloaded. (Optional)

### Software Requirements
Tested using Miniconda with Python 3.7 and the package versions detailed in dh_nature.yml. A new environment can be created with the yaml file by running ```conda env create -f dh_nature.yml```. Linux and macOS currently supported.

### Input Requirements
Input files must be DICOM files with a SOP Class UID of either 1.2.840.10008.5.1.4.1.1.1.2 or 1.2.840.10008.5.1.4.1.1.13.1.3. Additionally, the Burned In Annotation tag value must equal 'NO' (in case there are annotations that could contain PHI).

The final pre-processed zip file sent to the evaluation server cannot exceed 2GB. If many large studies are being evaluated, it is advised to break up the studies into several runs to facilitate more efficient processing.

### Output Description
Upon successful evaluation completion, a study-level csv and a file-level csv will be returned. The study-level csv contains a row for each successfully evaluated study with a 0-1 AI score for each study, where a higher score indicates higher suspicion of malignancy. The file-level csv contains a row for each successfully evaluated file with a score for each file, and for files that have a score above a certain threshold, coordinates for the top scoring bounding box for the file. The ```--plot_images``` parameter can be used to display the bounding boxes.
