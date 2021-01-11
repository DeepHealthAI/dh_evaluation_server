import os
import cv2
import pydicom
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def plot_and_save_ims(bbox_df, output_dir):
    """
    Plots and saved predicted bounding boxes and scores on dicom images
    :param bbox_df: DataFrame. A DataFrame with the columns ['file_path', 'x1', 'y1', 'x2', 'y2', 'score', 'slice'],
    where file_path is the absolute path to the dicom file
    :param output_dir: str. The absolute path to the output directory to which the images will be saved
    :return: None
    """
    assert all([col in bbox_df.columns for col in ['file_path', 'x1', 'y1', 'x2', 'y2', 'score', 'slice']]), \
        'The expected columns in the bbox DF were not all found'
    files = bbox_df[bbox_df['score'].notnull()].groupby('file_path').apply(lambda df: df.loc[df['score'].idxmax()])
    for index, row in files.iterrows():
        if row.isnull().any():
            # No predictions available for this image
            continue
        im = load_im(row['file_path'], int(row['slice']))
        im = plot_box(im, row['x1'], row['y1'], row['x2'], row['y2'], row['score'])
        path_head, fname = os.path.split(row['file_path'])
        _, file_dirname = os.path.split(path_head)
        if not os.path.isdir(os.path.join(output_dir, file_dirname)):
            os.makedirs(os.path.join(output_dir, file_dirname))
        Image.fromarray(im).save(os.path.join(output_dir, file_dirname, fname + '_plot.png'))


def plot_box(im_orig, x1, y1, x2, y2, score, thickness=3, box_color=(255, 0, 0), text_color=(255, 0, 0), font_scale=3.5, font_thickness=4, line_type=2):
    """
    Draws the bounding box and score associated with the passed-in image on the image.
    :param im_orig: numpy array. A 2D array of dtype uint8
    :param x1: int or float. The X-coordinate of the top left corner of the bounding box
    :param y1: int or float. The Y-coordinate of the top left corner of the bounding box
    :param x2: int or float. The X-coordinate of the bottom right corner of the bounding box
    :param y2: int or float. The Y-coordinate of the bottom right corner of the bounding box
    :param score: int or float. The predicted malignancy score
    :param thickness: int. The thickness of the edge of the drawn rectangle (optional)
    :param box_color: tuple. The color of the drawn rectangle (optional)
    :param text_color: tuple. The color of the drawn score text (optional)
    :param font_scale: int or float. The scale of the drawn text (optional)
    :param font_thickness: int or float. The thickness of the line used to draw the text (optional)
    :param line_type: int. lineType argument for cv2.putText (see documentation for lineType details) (optional)
    :return: numpy array. 3-channel numpy array with the bounding box and score text drawn
    """
    assert isinstance(im_orig, np.ndarray), 'Expected type ndarray for the image, got type {}'.format(type(im_orig))
    assert all([isinstance(coord, int) or isinstance(coord, float) for coord in [x1, y1, x2, y2]]), 'Expected all coordinates to have type int or float'
    assert isinstance(score, int) or isinstance(score, float), 'Expected score to have type int or float, got type {}'.format(type(score))
    assert 0 <= score <= 1, 'Expected score to be in the range [0, 1], got {}'.format(score)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    assert x1 < x2, 'Expected x1 < x2, got x1: {}, x2: {}'.format(x1, x2)
    assert y1 < y2, 'Expected y1 < y2, got y1: {}, y2: {}'.format(y1, y2)
    im = im_orig.copy()
    assert len(im.shape) == 2, 'Expected an image with two axes, got shape {}'.format(im.shape)
    im = np.stack([im, im, im], axis=-1)
    cv2.rectangle(im, (x1, y1), (x2, y2), thickness=thickness, color=box_color)
    cv2.putText(im, 'Score: {:.4f}'.format(score), org=(35, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=text_color, lineType=line_type, bottomLeftOrigin=False,
                thickness=font_thickness)

    return im


def load_im(file_path, slice_num):
    """
    Loads the pixel array from the specified dicom file.
    :param file_path: str. The absolute path to the dicom file
    :param slice_num: int. The number of the slice that the box is found on
    :return: numpy array. 2D loaded pixel array of dtype uint8
    """
    ds = pydicom.dcmread(file_path)
    assert isinstance(slice_num, int), 'Expected slice_num to be an int, got type {}'.format(type(slice_num))

    # When slice_number = -1 but file is DBT, set slice_num to be middle slice
    if ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.13.1.3':
        if slice_num == -1:
            num_frames = int(getattr(ds, 'NumberOfFrames', -1))
            # If NumberOfFrames is not a field in DBT, try to infer from pixel_array
            if num_frames == -1:
                im = ds.pixel_array
                if len(im.shape) != 3:
                    raise Exception('DBT file contains 2 dimensional pixel_array')
                num_frames = im.shape[0]
            slice_num = int(num_frames / 2)
        im = ds.pixel_array
        im = pydicom.pixel_data_handlers.apply_voi_lut(im, ds)[slice_num, :, :]

    elif ds.SOPClassUID == '1.2.840.10008.5.1.4.1.1.1.2':
        im = ds.pixel_array
        im = pydicom.pixel_data_handlers.apply_voi_lut(im, ds)

    else:
        raise Exception('DICOM at {} is not DXM or DBT file.'.format(file_path, ds.SOPClassUID))

    return bytescale(im)


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    COPIED FROM SCIPY SOURCE CODE, NOW DEPRECATED
    """
    if data.dtype == np.uint8:
        return data

    if high < low:
        raise ValueError("`high` should be larger than `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data * 1.0 - cmin) * scale + 0.4999
    bytedata[bytedata > high] = high
    bytedata[bytedata < 0] = 0

    return np.cast[np.uint8](bytedata) + np.cast[np.uint8](low)


def parse_args():
    parser = argparse.ArgumentParser(description='Plotting paths parser')
    parser.add_argument('--bbox_df', required=True, type=str,
                        help='The path to the bounding box CSV containing the columns'
                              ' ["file_path", "x1", "y1", "x2", "y2", "score", "slice"')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='The path to the top-level directory that the plotted images will be saved to. '
                             'One subdirectory will be created inside this directory for each initial subdirectory '
                             'containing dicom files as specified by the "file_path" column in the bounding box CSV')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        bbox_df = pd.read_csv(args.bbox_df)
        plot_and_save_ims(bbox_df=bbox_df, output_dir=args.output_dir)
    except FileNotFoundError:
        print('{} was not found. Please make sure that you have entered the full (absolute) path to the CSV.'.format(args.bbox_df))