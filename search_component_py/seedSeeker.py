import feature
import os
import numpy as np
from fanCalcLib import formMinComb_c
import generic
import rule
import tqdm
import random
from multiprocessing import Pool
import re


def tileWallGen(seed, raw_list):
    expanded_list = [i for i in raw_list for _ in range(4)]
    for i in range(1, 136):
        seed = (seed * 69069 + 1) % 4194304
        idx = seed % i
        x = expanded_list[i]
        expanded_list[i] = expanded_list[idx]
        expanded_list[idx] = x
    return expanded_list


def isValid(tileWall, tile_list, up_cut_threshold=-1, strict_threshold=[]):
    if up_cut_threshold == -1 and strict_threshold == []:
        return True
    hand_list = []
    init_dist = []
    for i in range(4):
        hand_list.append(tileWall[34 * (i + 1) - 1 : 34 * (i + 1) - 14 : -1])
    for hand in hand_list:
        hand_enc = rule.from_canonical_to_custom_encoding(hand)
        tile_list_cp = rule.update_tile_info(tile_list, hand_enc)
        (
            list1,
            list1id,
            list2,
            list2id,
            list3,
            list3id,
            list4,
            list4id,
        ) = formMinComb_c(hand_enc, [], tile_list_cp, 0, 1, 15, 7)
        list_comp = []
        if list1id != -1:
            list_comp.append(list1[list1id])
        if list2id != -1:
            list_comp.append(list2[list2id])
        if list3id != -1:
            list_comp.append(list3[list3id])
        if list4id != -1:
            list_comp.append(list4[list4id])
        min_dist = 9
        for entry in list_comp:
            if entry[0] < min_dist:
                min_dist = entry[0]
        min_dist -= 1  # adjust from dist to hu to dist to 上听

        if up_cut_threshold != -1:
            if min_dist < up_cut_threshold:
                # print("At least one init dist is {}", min_dist)
                return False
        init_dist.append(min_dist)
    if strict_threshold != []:
        for thresh in strict_threshold:
            if list(init_dist) == [thresh, thresh, thresh, thresh]:
                # print("At least one init dist is {}", min_dist)
                return True

    return False


def seedSeeking(seed_num, up_cut_threshold=-1, strict_threshold=-1):
    nameRand = random.randint(0, 100000000)
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
    seed_list = []
    while len(seed_list) < seed_num:
        seed = random.randint(0, 100000000)
        wall = tileWallGen(seed, tile_list_raw)
        if isValid(wall, tile_list, up_cut_threshold, strict_threshold):
            seed_list.append(seed)
    with open(
        "seedInfo_c{}_s{}_{}.txt".format(up_cut_threshold, strict_threshold, nameRand),
        "w",
    ) as f:
        f.write(str(seed_list))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 2
    pool = Pool(cpuCount)
    for _ in range(54):
        # for fil in file_unprocessed:
        pool.apply_async(
            seedSeeking,
            args=(700, -1, [3, 4]),
        )
    pool.close()
    pool.join()
    keyDict = {}
    dir_list = os.listdir(".")
    seedInfoList = []
    keyDict = {}
    for file in dir_list:
        if file[:12] == "seedInfo_c-1":
            seedInfoList.append(file)
    for file in seedInfoList:
        with open(file, "r") as f:
            a = f.readline()
            a = re.sub("[^0-9]+", " ", a)
            b = a.split()
            for i in b:
                keyDict[int(i)] = 1
    keyList = keyDict.keys()
    with open("all_seed_c-1_s[3,4].txt", "w") as f:
        for k in keyList:
            f.write(str(k) + "\n")
