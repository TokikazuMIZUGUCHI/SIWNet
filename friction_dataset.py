import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class FrictionDataset(Dataset):
    """
    Implementation of the dataset used in the study.
    Loads image paths and grip factors from CSV files and prepares them for use in training.
    """
    def __init__(self, data_paths, transform):
        """
        Initializes the dataset by loading data from the provided CSV files.

        :param data_paths: List of paths to CSV files containing the data.
        :param transform: PyTorch transforms to apply to the images.
        """
        # Load data from CSV files
        data_list = [pd.read_csv(path) for path in data_paths]  
        self.data = pd.concat(data_list).reset_index(drop=True)  # Reset index to ensure continuous indexing
        self.data_len = len(self.data.index)
        self.img_path = np.asarray(self.data.iloc[:, 0])

        # Convert grip factor from string to float
        self.grip = self.data.iloc[:, 1].astype(float)
        
        # Scale grip factor to friction factor (0...1) for normalisation
        self.friction_factor = (self.grip - 0.09) / (0.82 - 0.09)
 
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Retrieves an item from the dataset at the specified index.

        :param index: Index of the item to retrieve.
        :return: Tuple containing the transformed image and its corresponding friction factor.
        """
        try:
            img_path = self.img_path[index]
            img = Image.open(img_path).convert("RGB")
            # Get friction factor
            ff = self.friction_factor[index]
            ff = np.float32(ff)
            # Transform image 
            img = self.transform(img)
            
            return (img, ff)
        except KeyError as e:
            print(f"Index {index} is out of bounds. Error: {e}")
            raise
        except Exception as e:
            print(f"An error occurred while processing index {index}: {e}")
            raise

    def __len__(self):
        """
        Returns the total number of items in the dataset.

        :return: Length of the dataset.
        """
        return self.data_len
