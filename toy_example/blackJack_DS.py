from torch.utils.data import Dataset
import numpy as np
import os
import json
from multiprocessing import Pool
import torch
from collections import defaultdict
import copy


class blackJackDataset(Dataset):
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
        raw_state = self.data[index]
        (dealer_first, ace_usable, total_value) = raw_state
        transformed_data = [
            ace_usable and dealer_first == 1,
            ace_usable and dealer_first == 2,
            ace_usable and dealer_first == 3,
            ace_usable and dealer_first == 4,
            ace_usable and dealer_first == 5,
            ace_usable and dealer_first == 6,
            ace_usable and dealer_first == 7,
            ace_usable and dealer_first == 8,
            ace_usable and dealer_first == 9,
            ace_usable and dealer_first == 10,
            (not ace_usable) and dealer_first == 1,
            (not ace_usable) and dealer_first == 2,
            (not ace_usable) and dealer_first == 3,
            (not ace_usable) and dealer_first == 4,
            (not ace_usable) and dealer_first == 5,
            (not ace_usable) and dealer_first == 6,
            (not ace_usable) and dealer_first == 7,
            (not ace_usable) and dealer_first == 8,
            (not ace_usable) and dealer_first == 9,
            (not ace_usable) and dealer_first == 10,
        ]
        return (
            1.0 * torch.tensor(transformed_data),
            torch.tensor(total_value),
            torch.tensor(1.) if self.label[index] > 0 else torch.tensor(0.),
            # torch.tensor(1.0 * self.label[index]),
        )


if __name__ == "__main__":
    ds = blackJackDataset("./bj_opt_data.json")
    print(len(ds))
    print(ds[1])
