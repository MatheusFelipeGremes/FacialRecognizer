"""
Module containing a dataset class for Labeled Faces in the Wild (LFW) dataset.

Labeled Faces in the Wild (LFW) dataset is a collection of pairs of images with labels indicating
whether the images depict the same person (match) or different persons (mismatch). This dataset
is commonly used for face recognition tasks.

More information about the dataset can be found in the readme file available at:
https://vis-www.cs.umass.edu/lfw/README.txt

This module provides a dataset class, `LabeledFacesWildDataset`, for loading and working with the LFW dataset.

Classes:
    LabeledFacesWildDataset: A dataset class for loading LFW dataset.

Usage:
    To use this module, import the `LabeledFacesWildDataset` class and create an instance by providing
    the path to the directory containing the images and the path to the annotations file.
    For example:

    ```
    from facial_recognizer.dataset import LabeledFacesWildDataset

    annotations_file = 'dataset/pairsDevTrain.txt'
    img_dir = 'dataset/lfw_funneled'

    dataset = LabeledFacesWildDataset(img_dir, annotations_file)
    ```

    Once the dataset is instantiated, it can be used like any other PyTorch Dataset object.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

if TYPE_CHECKING:
    from torch import Tensor


class LabeledFacesWildDataset(Dataset):
    """
    Dataset class for Labeled Faces in the Wild (LFW) dataset.

    Readme of the dataset: https://vis-www.cs.umass.edu/lfw/README.txt

    This dataset contains pairs of images along with labels indicating whether the images are of the same person
    (match) or different persons (mismatch). The dataset is divided into matched pairs and mismatched pairs.

    Args:
        img_dir (str): Path to the directory containing the images.
        annotations_file (str): Path to the file containing annotations specifying pairs of images and their labels.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
                                         Default is None.

    Attributes
    ----------
        df_match (DataFrame): DataFrame containing information about matched pairs of images.
        df_miss_match (DataFrame): DataFrame containing information about mismatched pairs of images.
        img_dir (str): Path to the directory containing the images.
        transform (callable): A function/transform to apply to the images.

    Methods
    -------
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the given index.

    Examples
    --------
        dataset = LabeledFacesWildDataset(img_dir='lfw_funneled',
                                          annotations_file='pairsDevTrain.txt',
                                          transform=transforms.Compose([transforms.Resize(256),
                                                                        transforms.CenterCrop(224),
                                                                        transforms.ToTensor()]))
    """

    def __init__(self, img_dir: str, annotations_file: str, transform: None = None):
        """
        Initialize the LabeledFacesWildDataset object.

        Args:
            img_dir (str): Path to the directory containing the images.
            annotations_file (str): Path to the file containing annotations specifying pairs of images and their labels.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
                                             Default is None.
        """
        with Path(annotations_file).open() as file:
            num_lines = int(next(file))

        self.df_match = pd.read_csv(annotations_file, sep='\t', skiprows=1, nrows=num_lines, header=None)
        self.df_miss_match = pd.read_csv(annotations_file, sep='\t', skiprows=num_lines + 1, header=None)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns
        -------
            int: Total number of samples.
        """
        return len(self.df_match) + len(self.df_miss_match)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Retrieve the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns
        -------
            Tuple: A tuple containing images.

        This method retrieves a sample from the dataset at the given index.
        It returns a tuple containing four images:
        - The first two images represent a match.
        - The last two images represent a mismatch.
        """
        match_images = self._get_match_images(idx)
        mismatch_images = self._get_mismatch_images(idx)
        return match_images[0], match_images[1], mismatch_images[0], mismatch_images[0]

    def _get_match_images(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve a pair of images representing a match at the given index.

        Args:
            idx (int): Index of the match pair to retrieve.

        Returns
        -------
            tuple: A tuple containing two images representing a match.

        This method retrieves a pair of images representing a match from the dataset at the given index.
        It returns a tuple containing two images. The first image corresponds to the first image in the match pair,
        and the second image corresponds to the second image in the match pair.
        """
        match_1_name = self.df_match.iloc[idx, 0]
        match_1_img_number = self.df_match.iloc[idx, 1]
        match_2_img_number = self.df_match.iloc[idx, 2]

        match_1_img_path = Path(self.img_dir) / match_1_name / f'{match_1_name}_{match_1_img_number:04d}.jpg'
        match_2_img_path = Path(self.img_dir) / match_1_name / f'{match_1_name}_{match_2_img_number:04d}.jpg'

        image_1 = read_image(str(match_1_img_path))
        image_2 = read_image(str(match_2_img_path))

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2

    def _get_mismatch_images(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retrieve a pair of images representing a mismatch at the given index.

        Args:
            idx (int): Index of the mismatch pair to retrieve.

        Returns
        -------
            tuple: A tuple containing two images representing a mismatch.

        This method retrieves a pair of images representing a mismatch from the dataset at the given index.
        It returns a tuple containing two images. The first image corresponds to one image in the mismatch pair,
        and the second image corresponds to the other image in the mismatch pair.
        """
        mismatch_1_name = self.df_miss_match.iloc[idx, 0]
        mismatch_2_name = self.df_miss_match.iloc[idx, 2]
        mismatch_1_img_number = self.df_miss_match.iloc[idx, 1]
        mismatch_2_img_number = self.df_miss_match.iloc[idx, 3]

        mismatch_1_img_path = Path(self.img_dir) / mismatch_1_name / f'{mismatch_1_name}_{mismatch_1_img_number:04d}.jpg'
        mismatch_2_img_path = Path(self.img_dir) / mismatch_2_name / f'{mismatch_2_name}_{mismatch_2_img_number:04d}.jpg'

        image_1 = read_image(str(mismatch_1_img_path))
        image_2 = read_image(str(mismatch_2_img_path))

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


def main() -> int:
    """
    Test function for LabeledFacesWildDataset.

    Note
    ----
    This function is intended for testing purposes only.
    """
    annotations_file = r'dataset\pairsDevTrain.txt'
    img_dir = r'dataset\lfw_funneled'

    dataset = LabeledFacesWildDataset(img_dir, annotations_file)

    image_1, image_2, image_3, image_4 = dataset[0]

    return 0
