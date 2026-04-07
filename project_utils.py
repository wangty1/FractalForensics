import json
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms
from PIL import Image


class JsonConfig:
    """ Load config files from json. """

    def __init__(self):
        self.__json__ = None

    def load_json_file(self, path):
        with open(path, 'r') as file:
            self.__json__ = json.load(file)
            file.close()

        self.set_items()

    def load_json(self, json_dict):
        self.__json__ = json_dict

        self.set_items()

    def set_items(self):
        for key in self.__json__:
            self.__setattr__(key, self.__json__[key])

    def get_items(self):
        items = []
        for key in self.__json__:
            items.append((key, self.__json__[key]))
        return items

    def write_to_log(self, log_path, filename):
        with open(os.path.join(log_path, filename), 'w') as f:
            f.write('Training begins at ' + str(datetime.datetime.now()) + '.\n')
            f.write("=== Configuration Settings ===\n")
            for key, value in self.__json__.items():
                f.write(f"{key}: {value}\n")
            f.write("=== End of Config ===\n\n")


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class ImageDataset(Dataset):
    """Loads images from a directory."""

    def __init__(self, img_path, img_size, mode='train'):
        self.img_path = img_path
        self.img_size = img_size
        self.img_files = sorted(os.listdir(os.path.join(img_path)))
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((int(self.img_size * 1.1), int(self.img_size * 1.1))),
                transforms.RandomCrop((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_fname = os.path.join(self.img_path, self.img_files[idx])
        img = Image.open(img_fname).convert('RGB')
        img = self.transform(img)
        return img, self.img_files[idx]


class WatermarkDataset(Dataset):
    """Loads watermarks from precomputed .npy file."""

    def __init__(self, watermark_path):
        self.watermarks = np.load(watermark_path, allow_pickle=True)

    def __len__(self):
        return len(self.watermarks)

    def __getitem__(self, idx):
        decimal_wtm = torch.tensor(self.watermarks[idx], dtype=torch.long)
        binary_wtm = ((decimal_wtm.unsqueeze(0) >> torch.arange(3, -1, -1).view(4, 1, 1)) & 1).float()
        return binary_wtm


def make_loader(configs, mode='train', use_random=True):
    img_path = os.path.join(configs.img_path, mode)
    img_dataset = ImageDataset(img_path, configs.img_size, mode)
    img_loader = DataLoader(img_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=8, drop_last=True)
    if use_random:
        return img_loader
    else:
        reso = str(configs.img_size) + '_' + str(configs.wtm_size)
        watermark_path = os.path.join(configs.watermark_path, reso, mode, 'watermarks_' + mode + '.npy')
        wtm_dataset = WatermarkDataset(watermark_path)
        wtm_loader = DataLoader(wtm_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=8, drop_last=True)
        return img_loader, wtm_loader


def norm(tensor):
    return (tensor - 0.5) / 0.5


def denorm(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, min=0, max=1)


def wtm_error_rate(wtms, wtms_recover):
    """ Bit-wise error rate. """
    total_bits = wtms.numel()
    gt_bits = wtms >= 0.5
    recovered_bits = wtms_recover >= 0.5
    error_bits = (gt_bits != recovered_bits).sum().item()
    error_rate = error_bits / total_bits
    return error_rate


def patch_accuracy(wtms, wtms_recover):
    """ Patch-wise error rate. """
    gt_bits = wtms >= 0.5
    recovered_bits = wtms_recover >= 0.5
    # Check if all 4 bits match per patch -> (B, wtm_size, wtm_size)
    patch_match = (gt_bits == recovered_bits).all(dim=1)
    correct_patches = patch_match.sum().item()
    total_patches = patch_match.numel()
    accuracy = correct_patches / total_patches
    return accuracy


def patch_accuracy_per_image(wtms, wtms_recover):
    """ Compute patch-wise recovery accuracy for each image in the batch for AUC score computation. """
    gt_bits = wtms >= 0.5
    recovered_bits = wtms_recover >= 0.5

    # (B, 4, H, W) → check all 4 bits per patch → (B, H, W)
    patch_match = (gt_bits == recovered_bits).all(dim=1)

    # Compute per-image accuracy: number of correct patches / total patches
    per_image_acc = patch_match.view(patch_match.size(0), -1).float().mean(dim=1)  # Shape: (B,)
    return per_image_acc.cpu().numpy()  # Return as NumPy array for AUC scoring
