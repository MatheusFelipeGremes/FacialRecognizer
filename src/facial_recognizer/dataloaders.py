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

    annotations_file = 'dataset/lfw_funneled/pairs_01.txt'
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
        df_tripled (DataFrame): DataFrame containing information about triplets of images.
        img_dir (str): Path to the directory containing the images.
        transform (callable): A function/transform to apply to the images.

    Methods
    -------
        __len__(): Returns the total number of samples in the dataset.
        __getitem__(idx): Retrieves the sample at the given index.

    Examples
    --------
        # Example of how the data is formatted in the annotations_file:
        # Each line contains paths to four images, representing a triplet
        # The first two images are matches, and the last two images are mismatches
        # Example:

        # Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0001.jpg
        # Abdel_Nasser_Assidi/Abdel_Nasser_Assidi_0002.jpg
        # Aaron_Tippin/Aaron_Tippin_0001.jpg
        # Enos_Slaughter/Enos_Slaughter_0001.jpg
        #
        # Ai_Sugiyama/Ai_Sugiyama_0001.jpg
        # Ai_Sugiyama/Ai_Sugiyama_0002.jpg
        # Aaron_Tippin/Aaron_Tippin_0001.jpg
        # Juan_Carlos_Ortega/Juan_Carlos_Ortega_0001.jpg
        #
        # Ai_Sugiyama/Ai_Sugiyama_0001.jpg
        # Ai_Sugiyama/Ai_Sugiyama_0004.jpg
        # Aaron_Tippin/Aaron_Tippin_0001.jpg
        # Marlon_Devonish/Marlon_Devonish_0001.jpg

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
            data = [line.strip() for line in file if line.strip()]

        data_split = [data[i : i + 4] for i in range(0, len(data), 4)]

        self.df_tripled = pd.DataFrame(data_split)
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
        match_1_img_path = Path(self.img_dir) / Path(self.df_tripled.iloc[idx, 0])
        match_2_img_path = Path(self.img_dir) / Path(self.df_tripled.iloc[idx, 1])
        mismatch_1_img_path = Path(self.img_dir) / Path(self.df_tripled.iloc[idx, 2])
        mismatch_2_img_path = Path(self.img_dir) / Path(self.df_tripled.iloc[idx, 3])

        match_image_1 = read_image(str(match_1_img_path))
        match_image_2 = read_image(str(match_2_img_path))
        mismatch_image_1 = read_image(str(mismatch_1_img_path))
        mismatch_image_2 = read_image(str(mismatch_2_img_path))

        if self.transform:
            match_image_1 = self.transform(match_image_1)
            match_image_2 = self.transform(match_image_2)
            mismatch_image_1 = self.transform(mismatch_image_1)
            mismatch_image_2 = self.transform(mismatch_image_2)

        return match_image_1, match_image_2, mismatch_image_1, mismatch_image_2


def main() -> int:
    """
    Test function for LabeledFacesWildDataset.

    Note
    ----
    This function is intended for testing purposes only.
    """
    import torchvision.transforms.functional as TF

    img_dir = r'archive/lfw-funneled/lfw_funneled'
    pairs_txt = r'archive/lfw-funneled/lfw_funneled/pairs_01.txt'

    dataset = LabeledFacesWildDataset(img_dir, pairs_txt)

    image_1, image_2, image_3, image_4 = dataset[0]

    # Convertendo o tensor para uma imagem PIL
    image_pil_1 = TF.to_pil_image(image_1)
    image_pil_2 = TF.to_pil_image(image_2)
    image_pil_3 = TF.to_pil_image(image_3)
    image_pil_4 = TF.to_pil_image(image_4)

    # Visualizando a imagem
    image_pil_1.save('visualizar_image_1.png')
    image_pil_2.save('visualizar_image_2.png')
    image_pil_3.save('visualizar_image_3.png')
    image_pil_4.save('visualizar_image_4.png')

    breakpoint()

    return 0
