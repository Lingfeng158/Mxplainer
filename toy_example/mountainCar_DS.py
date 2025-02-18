from torch.utils.data import Dataset
import numpy as np
import os
import json
from multiprocessing import Pool
import torch
from collections import defaultdict
import copy


class MountainCarDataset(Dataset):
    def __init__(self, path_to_data):
        with open("{}".format(path_to_data), "r") as f:
            f_read = json.load(f)
        self.data = f_read["state"]
        self.label = f_read["action"]

    def __len__(self):
        return len(self.label)

    def get(self, index):
        return self.data[index], self.label[index]

    def __getitem__(self, index):
        prep_data = self.data[index]
        transformed_data = 1 * [
            prep_data[0] <= 0 and prep_data[1] <= 0,
            prep_data[0] <= 0 and prep_data[1] > 0,
            prep_data[0] > 0 and prep_data[1] <= 0,
            prep_data[0] > 0 and prep_data[1] > 0,
        ]
        return torch.tensor(transformed_data), torch.tensor(1.0 * self.label[index])


if __name__ == "__main__":
    ds = MountainCarDataset("./opt_data.json")
    print(len(ds))
    print(ds.get(1))
