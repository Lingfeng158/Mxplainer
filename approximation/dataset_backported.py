from torch.utils.data import Dataset
import numpy as np
import os
from multiprocessing import Pool
import torch
from collections import defaultdict
import copy


def default_zero():
    return 0


def workload(path_to_data, file_name, is_training):
    cache = {
        "meta": [],
        "raw_meta_feature": [],
        "tile_wall_feature": [],
        "discard_history3": [],
        "pack_info3": [],
        "data_holder": [],
        "label_action": [],
        "label_tile": [],
        "verification_prep": [],
        "verification_label": [],
        "verification_info": [],
    }
    path_to_file = os.path.join(path_to_data, file_name)
    with open(path_to_file, "rb") as f:
        meta_list = np.load(f, allow_pickle=True)
        raw_meta_feature_list = np.load(f, allow_pickle=True)
        tile_wall_feature_list = np.load(f, allow_pickle=True)
        discard_history = np.load(f, allow_pickle=True)
        pack_info = np.load(f, allow_pickle=True)
        data_list = np.load(f, allow_pickle=True)
        label_action_list = np.load(f, allow_pickle=True)
        label_tile_list = np.load(f, allow_pickle=True)
        verification_data = np.load(f, allow_pickle=True)
        verification_label = np.load(f, allow_pickle=True)
        verification_info = np.load(f, allow_pickle=True)

        if len(meta_list) > 12:
            cache["meta"] = meta_list[12:]
            cache["raw_meta_feature"] = raw_meta_feature_list[12:]
            cache["tile_wall_feature"] = tile_wall_feature_list[12:]
            cache["discard_history3"] = discard_history[12:]
            cache["pack_info3"] = pack_info[12:]
            cache["data_holder"] = data_list[12:]
            cache["label_action"] = label_action_list[12:]
            cache["label_tile"] = label_tile_list[12:]
            cache["verification_prep"] = verification_data[12:]
            cache["verification_label"] = verification_label[12:]
            cache["verification_info"] = verification_info[12:]

    return cache


def data_split(data_obj):
    """
    Convert data_obj from 5*32*(34*4, 80, 34, 2) to (5*32*34*4, 5*32*80, 5*32*34, 5*32*2)
    """
    term1 = []
    term2 = []
    term3 = []
    term4 = []
    term5 = []
    for outer_layer in data_obj:
        # each outer_layer: 32*(...)
        term1_layer1 = []
        term2_layer1 = []
        term3_layer1 = []
        term4_layer1 = []
        term5_layer1 = []
        for inner_layer in outer_layer:
            # each inner_layer: (34*4, 80, 34, 2)
            term1_layer1.append(copy.deepcopy(inner_layer[0]))
            term2_layer1.append(copy.deepcopy(inner_layer[1]))
            term3_layer1.append(copy.deepcopy(inner_layer[2]))
            term4_layer1.append(copy.deepcopy(inner_layer[3]))
            term5_layer1.append(copy.deepcopy(inner_layer[4]))
        term1.append(copy.deepcopy(term1_layer1))
        term2.append(copy.deepcopy(term2_layer1))
        term3.append(copy.deepcopy(term3_layer1))
        term4.append(copy.deepcopy(term4_layer1))
        term5.append(copy.deepcopy(term5_layer1))
    return (
        np.array(term1),
        np.array(term2),
        np.array(term3),
        np.array(term4),
        np.array(term5),
    )


def mix_match_feature(discard_history, pack_info):
    return [
        discard_history[0],
        pack_info[0],
        discard_history[1],
        pack_info[1],
        discard_history[2],
        pack_info[2],
        discard_history[3],
        pack_info[3],
    ]


def decode_tile_wall(tile_wall_enc):
    """
    Split till_wall into tile 34* (-2, -1, 0, 1, 2)
    shape: 34*5
    """
    chi_able_list = [tile_wall_enc[:9], tile_wall_enc[9:18], tile_wall_enc[18:27]]
    chi_unable_list = tile_wall_enc[27:]
    return_list = []
    # for chi-able tile
    for tile_tensor in chi_able_list:
        enlarged_tensor = np.zeros(13)
        enlarged_tensor[2:11] = tile_tensor
        for i in range(2, 11):
            return_list.append(enlarged_tensor[i - 2 : i + 3])
    for tile in chi_unable_list:
        enlarged_tensor = np.zeros(5)
        enlarged_tensor[2] = tile
        return_list.append(enlarged_tensor)
    return_list_tensor = np.stack(return_list, axis=0)
    return return_list_tensor


def convert_info(v_info):
    file_no = int(v_info[0][:-4])
    return [file_no, int(v_info[1]), int(v_info[2])]


class MahjongSLDataset(Dataset):
    def __init__(self, path_to_data, begin=0, end=1, is_training=True):
        self.match_samples = os.listdir(path_to_data)
        self.total_matches = len(self.match_samples)  # total number of matches
        self.begin = int(begin * self.total_matches)  # start location by match
        self.end = int(end * self.total_matches)  # end location by match
        self.match_samples = self.match_samples[
            self.begin : self.end
        ]  # select by match
        self.matches = len(self.match_samples)

        self.cache = {
            "meta": [],
            "raw_meta_feature": [],
            "tile_wall_feature": [],
            "discard_history3": [],
            "pack_info3": [],
            "data_holder": [],
            "label_action": [],
            "label_tile": [],
            "verification_prep": [],
            "verification_label": [],
            "verification_info": [],
        }

        for f in self.match_samples:
            ret = workload(path_to_data, f, is_training)
            for k in ret:
                self.cache[k].extend(ret[k])

        self.samples = len(self.cache["meta"])

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        meta = self.cache["meta"][index]
        tile_wall_feature = self.cache["tile_wall_feature"][index]
        data = self.cache["data_holder"][index]
        data = data_split(data)

        # v4 change: add vector for fan summary
        fan = data[1][0]  # [5, 64, 80]
        fan_summary = np.sum(fan, axis=0)[:56]
        # normalize
        fan_max = np.max(fan_summary) + 1e-3
        fan_summary = fan_summary / fan_max

        discard_history = self.cache["discard_history3"][index]
        discard_history = np.array(discard_history)

        pack_info = self.cache["pack_info3"][index]
        pack_info = np.array(pack_info)

        ret = mix_match_feature(discard_history, pack_info)
        tile_wall_feature_holder = [tile_wall_feature]
        for ret_slice in ret:
            slice_feature = decode_tile_wall(ret_slice)
            tile_wall_feature_holder.append(slice_feature)
        tile_wall_feature_matrix = np.stack(tile_wall_feature_holder, axis=1)

        label_action = self.cache["label_action"][index]
        label_tile = self.cache["label_tile"][index]

        # v_data = self.cache['verification_prep'][index]
        # v_label = self.cache['verification_label'][index]
        v_info = self.cache["verification_info"][index]
        v_info = np.array(convert_info(v_info))

        # re-compile meta_feature
        remaining_tile_count, opponent_held_count = self.cache["raw_meta_feature"][
            index
        ]
        remaining_tile_count, opponent_held_count = (
            remaining_tile_count + 2,
            opponent_held_count + 2,
        )
        normalized_rtc, normalized_ohc = (
            remaining_tile_count / 120,
            opponent_held_count / 45,
        )

        meta_feature_new = np.array(
            [
                normalized_rtc,
                1 / remaining_tile_count,
                1 - normalized_rtc,
                normalized_ohc,
                1 / opponent_held_count,
                1 - normalized_ohc,
            ]
        )

        # v_info[0] = int(v_info[0].split(".")[0])
        # v_info = np.array(v_info).astype(int)
        (k, q, m, n, o) = data

        # v_info[0] = int(v_info[0].split(".")[0])
        # v_info = np.array(v_info).astype(int)
        return (
            torch.tensor(meta).int(),
            torch.tensor(meta_feature_new).float(),
            torch.tensor(tile_wall_feature_matrix).float(),
            torch.tensor(k).float(),
            torch.tensor(q).float(),
            torch.tensor(m).float(),
            torch.tensor(n).float(),
            torch.tensor(o).float(),
            # torch.tensor(fan_summary).float(),
            torch.tensor(label_action).float(),
            torch.tensor(label_tile).float(),
            torch.tensor(v_info).int(),
        )
