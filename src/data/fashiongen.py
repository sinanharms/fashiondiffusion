import h5py
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm


class FashionGenDataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")
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
            "input_image": torch.tensor(self.file["input_image"][idx]).permute(
                (2, 0, 1)
            ),
            "input_msrpUSD": self.file["input_msrpUSD"][idx][0],
            "input_name": self.file["input_name"][idx][0].decode("utf-8"),
            "input_pose": self.file["input_pose"][idx][0].decode("utf-8"),
            "input_productID": self.file["input_productID"][idx][0],
            "input_season": self.file["input_season"][idx][0].decode("utf-8"),
            "input_subcategory": self.file["input_subcategory"][idx][0].decode("utf-8"),
        }
        return sample
