import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
import src_py.feature

# from fanCalcLib import formMinComb_c
import src_py.rule
import numpy as np
from multiprocessing import Pool
import json


def default_zero():
    return 0


from collections import defaultdict


def load_from_result(dst, file):
    dst_path = os.path.join(dst, file)
    with open(dst_path, "rb") as f:
        # meta
        winner_id = np.load(f, allow_pickle=True)[0]
        fan_sum = np.load(f, allow_pickle=True)[0]
        score = np.load(f, allow_pickle=True)
        fan_list = np.load(f, allow_pickle=True)
    return (
        winner_id,  # 4
        fan_sum,
        score,
        fan_list,  # 5
    )


def workload(path, file):
    tmp_dict = defaultdict(default_zero)
    _, _, _, fan_list = load_from_result(path, file)
    for fan in fan_list:
        tmp_dict[fan] += 1
    return tmp_dict


if __name__ == "__main__":
    fan_distribution_dict = defaultdict(default_zero)

    cpuCount = os.cpu_count()
    path = "fan_statatistics"
    dir_list = os.listdir(path)
    # file_list1 = os.listdir(path)
    # file_list2 = os.listdir(dst)
    # file_unprocessed = list(set(file_list1) - set(file_list2))
    # postprocessing(path, dst, tile_list, "0.npy")
    ret_list = []
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        ret = pool.apply_async(
            workload,
            args=(path, fil),
        )
        ret_list.append(ret)
    pool.close()
    pool.join()
    for ret in ret_list:
        d = ret.get()
        for k in d.keys():
            fan_distribution_dict[k] += d[k]
    with open("statistics_general.json", "w", encoding="utf8") as f:
        json.dump(fan_distribution_dict, f, ensure_ascii=False, indent=2)
    print("Post Processing Done!")
