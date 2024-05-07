import os

import h5py
import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def load_fashiongen_data(data_dir: str):
    file = h5py.File(data_dir, "r")

    return file


def list_file_paths(data_dir: str):
    return (
        f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))
    )


class FashionGenDataset(Dataset):
    def __init__(self, data_dir: str):
        self.file = load_fashiongen_data(data_dir)
        self.length = len(self.file["index"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {
            "index": self.file["index"][idx][0],
            "index_2": self.file["index_2"][idx],
            "input_brand": self.file["input_brand"][idx][0].decode("utf-8"),
            "input_category": self.file["input_category"][idx][0].decode("utf-8"),
            "input_composition": self.file["input_composition"][idx][0].decode("utf-8"),
            "input_concat_description": self.file["input_concat_description"][idx][
                0
            ].decode("utf-8"),
            "input_department": self.file["input_department"][idx][0].decode("utf-8"),
            "input_description": self.file["input_description"][idx][0].decode("utf-8"),
            "input_gender": self.file["input_gender"][idx][0].decode("utf-8"),
            "input_image": T.ToPILImage()(
                torch.tensor(self.file["input_image"][idx]).permute((2, 0, 1))
            ),
            "input_msrpUSD": self.file["input_msrpUSD"][idx][0],
            "input_name": self.file["input_name"][idx][0].decode("utf-8"),
            "input_pose": self.file["input_pose"][idx][0].decode("utf-8"),
            "input_productID": self.file["input_productID"][idx][0],
            "input_season": self.file["input_season"][idx][0].decode("utf-8"),
            "input_subcategory": self.file["input_subcategory"][idx][0].decode("utf-8"),
        }
        return sample


class FashionGenDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_file, self.val_file = list_file_paths(data_dir)

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = FashionGenDataset(self.train_file)
            self.val_dataset = FashionGenDataset(self.val_file)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
