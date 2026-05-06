import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class CustomDataset(Dataset):
    def __init__(self, ds, t):
        self.ds = ds
        self.t = t

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.t(self.ds[idx]["jpg"].convert("RGB")), self.ds[idx]["cls"]

class FeatureDataset(Dataset):
    def __init__(self, features_dir):
        self.features_dir = features_dir

        self.features_files = np.load(os.path.join(self.features_dir, "imagenet256_features_all.npy"))
        self.labels_files = np.load(os.path.join(self.features_dir, "imagenet256_labels_all.npy"))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        return torch.from_numpy(feature_file), torch.from_numpy(label_file)
