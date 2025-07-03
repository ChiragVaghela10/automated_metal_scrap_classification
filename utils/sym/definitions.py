"""
This module contains constants which are used throughout the application (e.g. the root path of
this library)
"""
import os

PATH_ROOT, _ = os.path.split(os.path.dirname(os.path.dirname(__file__)))
PATH_DATA_DIRECTORY = os.path.join(PATH_ROOT, "datasets")

# Existing directory containing images of all angles
PATH_CONTAINER_IMAGES = os.path.join(PATH_DATA_DIRECTORY, 'container_images')

# New directory to store images of certain angle
PATH_IMAGES = os.path.join(PATH_DATA_DIRECTORY, 'sym_images')

camera_id_with_train = ['Pr8Ant', 'Pr9Ant', 'Pr12Ant', 'Pr8Pos', 'Pr9Pos', 'Pr13Pos', ]
camera_id_with_bucket = ['Par1CarA', 'Par2CarA', 'Par3CarA', 'Par1CarB', 'Par2CarB', 'Par3CarB']

ORIGINAL_DATASET = 'datasets/scrap_yard_original'
DENOISED_DATASET = 'datasets/scrap_yard_denoised'
HIST_EQUALIZED_DATASET = 'datasets/scrap_yard_hist_equalized'
PREPROCESSING_METHODS = ['DENOISE', 'HIST_EQUALIZE']
