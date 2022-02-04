import os
import re
from enum import Enum
from typing import Union

from tfrecord.torch.dataset import MultiTFRecordDataset
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from torch.utils.data import WeightedRandomSampler
import torchvision
from pathlib import Path

class DatasetType(Enum):
    TRAIN="train"
    VAL="validation"
    TEST="test"


def decode_image(data):
    # get BGR image from bytes
    np_image = np.array(Image.open(BytesIO(data["image/data"])))  # noqa
    data_transformed_float = np_image.astype(np.float32)
    data_normed = (data_transformed_float-127.5)/128
    #make channels first
    data_reordered = np.transpose(data_normed, (2, 0, 1))
    data["image/data"] = data_reordered

    return data


def transform_mnist(pil_image: Image.Image):
    converted = pil_image.convert("RGB")
    converted_array = np.array(converted.resize((224, 224)))   # noqa
    converted_array_reordered = np.transpose(converted_array, (2, 0, 1))
    converted_array_float = converted_array_reordered.astype(np.float32)
    data_normed = (converted_array_float-127.5)/128
    return data_normed


def create_dataset(data_base_folder: Union[Path,str], batch_size: int = 32, dataset_type: DatasetType = DatasetType.TRAIN) ->torch.utils.data.DataLoader:
    if isinstance(data_base_folder,str):
        data_base_folder = Path(data_base_folder)
    files_in_data_folder = os.listdir(data_base_folder)
    files_in_folder = list(filter(lambda x: dataset_type.value in x, files_in_data_folder))
    split_percentage = 1/len(files_in_folder)

    tfrecord_pattern = data_base_folder / (f"dataset_builder_basic_color-{dataset_type.value}"+".tfrecord-{}")
    #describes to what extent to sample from each tf record file
    splits = {}
    for current_file in files_in_folder:
        match = re.search(r"\d+-of-\d+", current_file)
        matched_string = current_file[match.start():match.end()]
        splits[matched_string] = split_percentage

    description = {"image/data": "byte", "class/pbk": "int", "class/base_color": "int"}
    dataset = MultiTFRecordDataset(str(tfrecord_pattern), None, splits,
                                   description=description, transform=decode_image, infinite=False, shuffle_queue_size=1024)
    return dataset


def create_loader(data_base_folder: Path, batch_size: int = 32, dataset_type: DatasetType = DatasetType.TRAIN, use_sampler_for_balancing=True) ->torch.utils.data.DataLoader:

    files_in_data_folder = os.listdir(data_base_folder)
    files_in_folder = list(filter(lambda x: dataset_type.value in x, files_in_data_folder))
    split_percentage = 1/len(files_in_folder)

    tfrecord_pattern = data_base_folder / (f"dataset_builder_basic_color-{dataset_type.value}"+".tfrecord-{}")
    #describes to what extent to sample from each tf record file
    splits = {}
    for current_file in files_in_folder:
        match = re.search(r"\d+-of-\d+", current_file)
        matched_string = current_file[match.start():match.end()]
        splits[matched_string] = split_percentage

    description = {"image/data": "byte", "class/pbk": "int", "class/base_color": "int"}
    dataset = MultiTFRecordDataset(str(tfrecord_pattern), None, splits,
                                   description=description, transform=decode_image, infinite=False, shuffle_queue_size=1024)

    '''
    if use_sampler_for_balancing:
        weights = [1,2,3,4,5]
        num_samples = 10
        sampler = WeightedRandomSampler(weights, num_samples, replacement=True, generator=None)   
    else:
        sampler = None     
    '''                    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=os.cpu_count())
    return loader


def create_loader_mnist():
    dataset = torchvision.datasets.MNIST(root=str(Config.GENERATED_CACHE_DIR), train=True, download=True, transform=transform_mnist)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=os.cpu_count())
    return loader

