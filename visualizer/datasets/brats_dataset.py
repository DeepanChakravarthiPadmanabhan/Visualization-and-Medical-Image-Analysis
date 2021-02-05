import os
import typing
import logging
import re
import gin
import torch
import nibabel as nib
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting as niplotting
import cv2

LOGGER = logging.getLogger(__name__)


@gin.configurable
class BraTSDataset(Dataset):
    """
    Multimodal Brain Tumor Segmentation dataset reader

    """

    def __init__(
        self,
        root_dir: str,
        transform: typing.Dict,
        input_width: int,
        input_height: int,
    ) -> None:

        self.root_dir = root_dir
        LOGGER.info("Root directory read: {}".format(root_dir))
        self.transform = transform
        self.directory_pattern = re.compile("/(BraTS19_[A-Z0-9_]+)")
        self.image_dir_list = self.get_doc_dirs_list(self.root_dir)
        LOGGER.info(
            "Number of image directories read: {}".format(len(self.image_dir_list))
        )

    def get_doc_dirs_list(self, root_dir: str) -> typing.List:
        """

        Args:
            root_dir: string
            Directory containing the nii.gz files

        Returns:
            image_dirs: list
            List containing directories of all the 4 modality nii.gz files

        """

        image_dirs = []

        for subdir, directory, files in os.walk(root_dir):
            image_directory = re.search(self.directory_pattern, subdir)
            if image_directory:
                image_dirs.append(subdir)

        if len(image_dirs) == 0:
            LOGGER.warning(
                "Root directory does not contain images. Kindly check the directory."
            )

        return image_dirs

    def get_image_label(self, directory):

        image_files = []

        for f in os.listdir(directory):
            full_path = os.path.join(directory, f)
            if os.path.isfile(full_path):
                image_files.append(full_path)

        for i in image_files:
            if "t1.npy" in i:
                t1 = np.load(i)

            if "t1ce.npy" in i:
                t1ce = np.load(i)

            if "t2.npy" in i:
                t2 = np.load(i)

            if "flair.npy" in i:
                flair = np.load(i)

            if "seg.npy" in i:
                seg = np.load(i)

        image = np.array([flair, t1, t1ce, t2])
        label = seg
        image = image.transpose(1, 2, 0)  # Channels/ Modalities to the last dimension

        return image, label

    def __len__(self):
        return len(self.image_dir_list)

    def __getitem__(self, idx: int) -> typing.Dict:

        directory = self.image_dir_list[idx]

        image, label = self.get_image_label(directory)

        sample = {
            "image": image,
            "label": label,
        }

        if self.transform:
            sample = self.transform(sample)

        # print('Final shape after transform: ', sample["image"].shape, sample["label"].shape)

        return sample
