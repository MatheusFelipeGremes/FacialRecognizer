from __future__ import annotations

import os

import numpy as np
import pytest
from facial_recognizer.dataloaders import LabeledFacesWildDataset
from PIL import Image


@pytest.fixture()
def create_image_directory(tmp_path):
    # Define the directory structure
    directory_structure = {
        'lfw-funneled': {
            'lfw_funneled': {
                'Aaron_Eckhart': ['Aaron_Eckhart_0001.jpg'],
                'Aaron_Guiel': ['Aaron_Guiel_0001.jpg'],
                'Aaron_Patterson': ['Aaron_Patterson_0001.jpg'],
                'Aaron_Peirsol': [
                    'Aaron_Peirsol_0001.jpg',
                    'Aaron_Peirsol_0002.jpg',
                    'Aaron_Peirsol_0003.jpg',
                    'Aaron_Peirsol_0004.jpg',
                ],
                'Aaron_Pena': ['Aaron_Pena_0001.jpg'],
                'Aaron_Sorkin': ['Aaron_Sorkin_0001.jpg', 'Aaron_Sorkin_0002.jpg'],
                'Aaron_Tippin': ['Aaron_Tippin_0001.jpg', 'Aaron_Tippin_0002.jpg'],
            },
        },
    }

    # Create the directory structure
    create_directory_structure(tmp_path, directory_structure)

    # Create txt file based on the directory structure
    txt_file_path = tmp_path / 'image_data.txt'
    with open(txt_file_path, 'w') as f:
        for subdir, subdirs_content in directory_structure.items():
            for subsubdir, subsubdirs_content in subdirs_content.items():
                for person, images in subsubdirs_content.items():
                    for image in images:
                        f.write(os.path.join(subdir, subsubdir, person, image) + '\n')

    # Return paths to the directory of images and the txt file
    return tmp_path / 'lfw-funneled', txt_file_path


def create_directory_structure(base_dir, structure):
    for directory, contents in structure.items():
        current_dir = base_dir / directory
        current_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory is created

        if isinstance(contents, dict):
            create_directory_structure(current_dir, contents)
        elif isinstance(contents, list):
            for file in contents:
                for i in range(10):  # Create 10 images per directory
                    img = np.random.randint(0, 256, size=(256, 256, 3), dtype=np.uint8)
                    with Image.fromarray(img) as image:
                        image.save(current_dir / f'{file[:-4]}_{i}.jpg')  # Save images with unique names


def test_create_image_directory(create_image_directory):
    images_dir, txt_file = create_image_directory
    assert os.path.exists(images_dir)
    assert os.path.exists(txt_file)


def test_LabeledFacesWildDataset(create_image_directory):
    images_dir, txt_file = create_image_directory
    dataset = LabeledFacesWildDataset(images_dir, txt_file)
    breakpoint()
