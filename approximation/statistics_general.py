import os
import sys

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
import src_py.feature

# from fanCalcLib import formMinComb_c
import src_py.rule
import numpy as np
from multiprocessing import Pool

# postprocess_general_v2: same as _general, but with updated fanCalcLib.so for faster processing


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


def postprocessing(path, dst, init_tile_list, file):
    dst_path = os.path.join(dst, file)
    (
        botzone_log,
        tileWall,
        pack,
        handWall,
        obsWall,
        remaining_tile,
        _,
        winner_id,
        wind,
        fan_sum,
        score,
        fan_list,
    ) = src_py.feature.load_log(path, file)

    with open(dst_path, "wb") as f:
        # meta
        np.save(f, [winner_id])
        np.save(f, [fan_sum])
        np.save(f, score)
        np.save(f, fan_list)


if __name__ == "__main__":
    tile_list = {}

    # fmt: off
    tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
                'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
                'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
                'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
    ]
    # fmt: on

    # 国标综合牌型, 自动搜索 + 算番库
    # 国标特殊牌型(8番以上)，特例搜索

    for tile in tile_list_raw:
        # tile_type = tile[0]
        # tile_rank = int(tile[1])
        tile_list[tile] = 4
    cpuCount = os.cpu_count()
    path = "data"
    dst = "fan_statatistics"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    dir_list = os.listdir(path)[:50000]
    # file_list1 = os.listdir(path)
    # file_list2 = os.listdir(dst)
    # file_unprocessed = list(set(file_list1) - set(file_list2))
    # postprocessing(path, dst, tile_list, "0.npy")
    pool = Pool(cpuCount)
    for fil in dir_list:
        # for fil in file_unprocessed:
        pool.apply_async(
            postprocessing,
            args=(path, dst, tile_list, fil),
        )
    pool.close()
    pool.join()
    print("Post Processing Done!")
