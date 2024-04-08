import albumentations

import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from albumentations.pytorch.transforms import ToTensorV2

class CIFAR10Data(Dataset):
    """Class for CIFAR10 Dataset
    """
    def __init__(self, dataset, transforms=None) -> None:
        """Constructor for CIFAR10Data

        Args:
            dataset (Object): CIFAR10 Data object
            transforms (Object, optional): Object to augment data
        """
        self.dataset = dataset
        self.transforms = transforms
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """Method to get image and lable of CIFAR10

        Args:
            index (int): Index of item required

        Returns:
            Tensor: Image and label in tensor format
        """
        image, label = self.dataset[index]

        image = np.array(image)

        if self.transforms:
            image = self.transforms(image=image)['image']

        return image, label

def _get_train_transforms():
    """Method to get train transform

    Returns:
        Object: Object to apply image augmentations
    """
    train_transforms = albumentations.Compose([
        # Add padding to image
        albumentations.PadIfNeeded(40, 40, always_apply=True), 
        # Randomly crop image
        albumentations.RandomCrop(32, 32),
        # Random cut out
        albumentations.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, 
                                     min_height=16, min_width=16,
                                     fill_value=(0.49139968, 0.48215841, 0.44653091), 
                                     mask_fill_value = None),
        # Normalize
        albumentations.Normalize([0.49139968, 0.48215841, 0.44653091], 
                                 [0.24703223, 0.24348513, 0.26158784]),
        # Convert to tensor
        ToTensorV2()])
    
    return train_transforms

def _get_test_transforms():
    """Method to get test transform

    Returns:
        Object: Object to apply image augmentations
    """
    test_transforms = albumentations.Compose(
        # Normalize
        [albumentations.Normalize([0.49139968, 0.48215841, 0.44653091], 
                                  [0.24703223, 0.24348513, 0.26158784]), 
        # Convert to tensor                          
        ToTensorV2()])
    return test_transforms

def _get_data(is_train, is_download):
    """Method to get data for training or testing

    Args:
        is_train (bool): True if data is for training else false
        is_download (bool): True to download dataset from iternet

    Returns:
        object: Oject of dataset
    """
    data = datasets.CIFAR10('../data', train=is_train, download=is_download)
    return data

def _get_data_loader(data, **kwargs):
    """Method to get data loader. 

    Args:
        data (object): Oject of dataset

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """
    loader = DataLoader(data, **kwargs)
    return loader

def get_train_data_loader(**kwargs):
    """Method to get data loader for trainig

    Args:
        batch_size (int): Number of images in a batch

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """
    train_transforms = _get_train_transforms()
    train_data = _get_data(is_train=True, is_download=True)
    train_data = CIFAR10Data(train_data, train_transforms)
    train_loader = _get_data_loader(data=train_data, **kwargs)
    return train_loader

def get_test_data_loader(**kwargs):
    """Method to get data loader for testing

    Args:
        batch_size (int): Number of images in a batch

    Returns:
        object: Object of DataLoader class used to feed data to neural network model
    """

    test_transforms = _get_test_transforms()
    test_data = _get_data(is_train=False, is_download=True)
    test_data = CIFAR10Data(test_data, test_transforms)
    test_loader = _get_data_loader(data=test_data, **kwargs)
    return test_loader