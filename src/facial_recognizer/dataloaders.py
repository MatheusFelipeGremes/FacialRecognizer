from __future__ import annotations

import os

import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


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
        Initializes the LabeledFacesWildDataset object.

        Args:
            img_dir (str): Path to the directory containing the images.
            annotations_file (str): Path to the file containing annotations specifying pairs of images and their labels.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
                                             Default is None.
        """
        with open(annotations_file) as file:
            num_lines = int(next(file))

        self.df_match = pd.read_csv(annotations_file, sep='\t', skiprows=1, nrows=num_lines, header=None)
        self.df_miss_match = pd.read_csv(annotations_file, sep='\t', skiprows=num_lines + 1, header=None)

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
            int: Total number of samples.
        """
        return len(self.df_match) + len(self.df_miss_match)

    def __getitem__(self, idx: int):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns
        -------
            Tuple: A tuple containing images.
        """
        match_1_name = self.df_match.iloc[idx, 0]
        match_1_img_number = self.df_match.iloc[idx, 1]
        match_2_img_number = self.df_match.iloc[idx, 2]

        match_1_img_path = os.path.join(self.img_dir, match_1_name, f'{match_1_name}_{match_1_img_number:04d}.jpg')
        match_2_img_path = os.path.join(self.img_dir, match_1_name, f'{match_1_name}_{match_2_img_number:04d}.jpg')

        image_1 = read_image(match_1_img_path)
        image_2 = read_image(match_2_img_path)

        mismatch_1_name = self.df_miss_match.iloc[idx, 0]
        mismatch_2_name = self.df_miss_match.iloc[idx, 2]
        mismatch_1_img_number = self.df_miss_match.iloc[idx, 1]
        mismatch_2_img_number = self.df_miss_match.iloc[idx, 3]

        mismatch_1_img_path = os.path.join(self.img_dir, mismatch_1_name, f'{mismatch_1_name}_{mismatch_1_img_number:04d}.jpg')
        mismatch_2_img_path = os.path.join(self.img_dir, mismatch_2_name, f'{mismatch_2_name}_{mismatch_2_img_number:04d}.jpg')

        image_3 = read_image(mismatch_1_img_path)
        image_4 = read_image(mismatch_2_img_path)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image_3 = self.transform(image_3)
            image_4 = self.transform(image_4)

        return image_1, image_2, image_3, image_4


def main():
    annotations_file = r'dataset\pairsDevTrain.txt'
    img_dir = r'dataset\lfw_funneled'

    dataset = LabeledFacesWildDataset(img_dir, annotations_file)

    breakpoint()

    image_1, image_2, image_3, image_4 = dataset[0]
