"""
Image preprocessing functions for Scrap Yard Manage project images
"""
import os
import cv2

from utils.sym.definitions import PATH_ROOT, ORIGINAL_DATASET, DENOISED_DATASET, HIST_EQUALIZED_DATASET, \
    PREPROCESSING_METHODS
from utils.augmentations import hist_equalize


class Preprocessing:
    """
    Preprocess the image dataset using image enhancement techniques before applying augmentation or YOLOv5 algorithm.
    """
    def __init__(self):
        self.src_sub_dirs = dict()
        self.dest_sub_dirs = dict()
        self.src_dirs_set = False
        self.dest_dirs_set = False
        self.method, self.src_dir, self.dest_dir = [None] * 3

    def set_source_dirs(self, src_dir=None):
        """
        Sets source directories for original image dataset to be processed
        """
        if src_dir:
            self.src_dir = os.path.join(os.path.dirname(PATH_ROOT), src_dir)
            src_train_dir = self.src_dir + '/images/train'

            # Source directory must have training dataset
            if os.path.exists(src_train_dir):
                self.src_sub_dirs['train'] = src_train_dir
            else:
                return print("Source directory must have 'images' folder with 'train' folder inside it.")

            src_val_dir = self.src_dir + '/images/val'
            self.src_sub_dirs['val'] = src_val_dir if os.path.exists(src_val_dir) else None
            self.src_dirs_set = True
        else:
            return print("Kindly, provide valid source directory path.")
        return self.src_dirs_set

    def set_target_dirs(self, dest_dir=None):
        """
        Set target directories for processed images
        """
        if dest_dir:
            self.dest_dir = os.path.join(os.path.dirname(PATH_ROOT), dest_dir)

            # Creates directory for processed training image dataset if not available
            dest_train_dir = self.dest_dir + '/images/train'
            if not os.path.exists(dest_train_dir):
                os.makedirs(dest_train_dir)
            self.dest_sub_dirs['train'] = dest_train_dir

            # Assign 'val' directory if available or create if required (source directory has validation subdirectory)
            # otherwise set None
            dest_val_dir = self.dest_dir + '/images/val'
            if os.path.exists(dest_val_dir):
                self.dest_sub_dirs['val'] = dest_val_dir
            elif self.src_dirs_set and self.src_sub_dirs['val']:
                os.makedirs(dest_val_dir)
                self.dest_sub_dirs['val'] = dest_val_dir
            else:
                self.dest_sub_dirs['val'] = None

            self.dest_dirs_set = True
        else:
            return print("Kindly, provide valid target directory path.")
        return self.dest_dirs_set

    def process(self, method=None):
        """
        Performing image enhancement techniques on all images of source dataset and creating same directory structure
        filled with processed image files. The destination directory will be created based on the preprocessing
        method used.
        """
        if not method:
            return "No preprocessing method specified."

        if self.src_dirs_set and method in PREPROCESSING_METHODS:
            if method == 'HIST_EQUALIZE':
                self.set_target_dirs(dest_dir=HIST_EQUALIZED_DATASET)
            elif method == 'DENOISE':
                self.set_target_dirs(dest_dir=DENOISED_DATASET)

            for dir_type, sub_dir in self.src_sub_dirs.items():
                training_images = os.listdir(sub_dir)
                img, processed_img = [None] * 2

                for filename in training_images:
                    if filename.__contains__('.jpg'):
                        img = cv2.imread(sub_dir + '/' + filename)

                    # Perform preprocessing operations according method parameter
                    if method == "HIST_EQUALIZE":
                        # Performing contrast limited adaptive histogram equalization (CLAHE) using in-built function
                        # in augmentations.py file
                        processed_img = hist_equalize(img, clahe=True) if img is not None else None
                    elif method == "DENOISE":
                        # Using recommended values of h, hForColorComponents, templateWindowSize, and searchWindowSize
                        processed_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

                    # Store the processed image to appropriate location
                    if processed_img is not None:
                        cv2.imwrite(os.path.join(self.dest_sub_dirs[dir_type], filename), processed_img)
                    else:
                        return print('Could not perform %s operation on %s' % method, filename)
        elif not self.src_dirs_set:
            return print("No directories defined for original image dataset.")
        elif method not in PREPROCESSING_METHODS:
            return print("Preprocessing method not found. Currently, supported options are: HIST_EQUALIZE, DENOISE")


preprocess = Preprocessing()
preprocess.set_source_dirs(src_dir=ORIGINAL_DATASET)
preprocess.process(method='DENOISE')
