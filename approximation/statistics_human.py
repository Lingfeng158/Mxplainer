import os
import json
from collections import defaultdict
import numpy as np
import feature_human as feature
import copy
import rule
from multiprocessing import Pool
import time

# from fanCalcLib import formMinComb_c

## preprocess_data_revised6_human.py from sl_v6


def default_zero():
    return 0


def default_list():
    return []


# fmt:off

tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
            'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
            ]
fan_list_raw = ["大四喜","大三元","绿一色","九莲宝灯","四杠","连七对","十三幺","清幺九","小四喜","小三元","字一色","四暗刻","一色双龙会",
"一色四同顺","一色四节高","一色四步高","三杠","混幺九","七对","七星不靠","全双刻","清一色","一色三同顺","一色三节高","全大",
    "全中","全小","清龙","三色双龙会","一色三步高","全带五","三同刻","三暗刻","全不靠","组合龙","大于五","小于五","三风刻","花龙",
    "推不倒","三色三同顺","三色三节高","无番和","妙手回春","海底捞月","杠上开花","抢杠和",
    "碰碰和","混一色","三色三步高","五门齐","全求人","双暗杠","双箭刻","全带幺",
    "不求人","双明杠","和绝张","箭刻","圈风刻","门风刻","门前清","平和","四归一",
    "双同刻","双暗刻","暗杠","断幺","一般高","喜相逢","连六","老少副","幺九刻",
    "明杠","缺一门","无字","边张","嵌张","单钓将","自摸"]

# fmt:on


def prepare_data(path_to_data, data_name, data_dst, meta_info_dict, target_id_list):
    (
        botzone_log,  # 1
        tileWall_log,  # 2
        pack_log,  # 3
        handWall_log,  # 4
        obsWall_log,  # 5
        remaining_tile_log,  # 6
        botzone_id,  # 7
        winner_id,  # 8
        prevailing_wind,  # 9
        fan_sum,  # 10
        score,
        fan_list,  # 11
    ) = feature.load_log(path_to_data, data_name)

    for target_id in target_id_list:
        # if entry does not contain target_id's operation
        if target_id not in meta_info_dict[botzone_id]:
            continue
        # else, find target_id's player_id within the round
        target_id_player_id = meta_info_dict[botzone_id].index(target_id)
        if target_id_player_id == winner_id:
            path_to_dst = os.path.join(data_dst + target_id, data_name)

            with open(path_to_dst, "wb") as f:
                # meta
                np.save(f, [winner_id])
                np.save(f, [fan_sum])
                np.save(f, score)
                np.save(f, fan_list)


if __name__ == "__main__":
    cpuCount = os.cpu_count()
    path = "human_data/data"
    dst = "fan_statatistics_"
    ref_path = "sl_prep_human_rev6_#sbot1"
    id_list = ["#sbot1"]
    for id in id_list:
        if not os.path.isdir(dst + id):
            os.makedirs(dst + id)

    with open("human_data/metadata.txt", "r") as f:
        meta_info = f.readlines()

    match_info_dict = defaultdict(default_list)
    for i in meta_info:
        entry = json.loads(i)
        players = entry["players"]
        match_info_dict[entry["match"]] = entry["players"]

    # file = "10743.npy"
    # prepare_data(path, file, dst)
    file_list = os.listdir(ref_path)
    file_list = sorted(file_list)
    pool = Pool(cpuCount)
    # start_time = time.time()
    for fil in file_list:
        pool.apply_async(
            prepare_data,
            args=(path, fil, dst, match_info_dict, id_list),
        )
        # print("Processing {}".format(fil))
        # prepare_data(path, fil, dst, match_info_dict, id_list)
        # print("Done with {}".format(fil))
        # end_time = time.time()
        # print("Time cost:{}".format(end_time - start_time))
        # start_time = end_time
    pool.close()
    pool.join()
    print("Post Processing Done!")
