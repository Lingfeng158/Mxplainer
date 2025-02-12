# file used to test drive
import sys
import os

sys.path.append("/data")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))
import numpy as np
import rule
import src_py.generic
from collections import defaultdict


# import json
# import src_py.rule
# import src_py.generic
# import random
# import src_py.special

# import MahjongGB
# from fanCalcLib import formMinComb_c
# from fanCalcLib import updateTileInfo_c

# from MahjongGB import MahjongFanCalculator
# from MahjongGB import MahjongShanten


def default_zero():
    return 0


def default_list():
    return []


import time

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

hand = {
    "T2": 1,
    "T4": 1,
    "F4": 2,
    "T3": 1,
    "T5": 1,
    "T6": 1,
    "T7": 1,
    "shown": [{"W1": 1, "W2": 1, "W3": 1}, {"B3": 1, "B4": 1, "B5": 1}],
}
hand1 = {"W3": 3, "J1": 2, "J2": 3, "B5": 1, "B6": 1, "B7": 1}
pack = [{"W9": 4, "AnGang": 1}]
# pack = []


def find_redundant_and_missing(hand_list, target_info):
    selected_tiles = target_info[1]
    target_tile_list = target_info[4]
    target_tile_dict = defaultdict(int)
    for d in target_tile_list:
        for t in d:
            target_tile_dict[t] += d[t]

    # find redundant tile from hand and selected_tiles
    redundant_tile = defaultdict(int)
    for t in hand_list:
        if t not in selected_tiles:
            selected_tiles[t] = 0
        if hand_list[t] - selected_tiles[t] > 0:
            redundant_tile[t] += hand_list[t] - selected_tiles[t]

    # find missing tile from selected_tiles and target_tile_dict
    missing_tile = defaultdict(int)
    for t in target_tile_dict:
        if target_tile_dict[t] - selected_tiles[t] > 0:
            missing_tile[t] = target_tile_dict[t] - selected_tiles[t]

    return redundant_tile, missing_tile


tile_list = updateTileInfo_c(tile_list, hand1, pack)
tile_list = {
    "B1": 4,
    "B2": 3,
    "B3": 3,
    "B4": 4,
    "B5": 4,
    "B6": 4,
    "B7": 3,
    "B8": 4,
    "B9": 3,
    "W1": 4,
    "W2": 4,
    "W3": 3,
    "W4": 3,
    "W5": 2,
    "W6": 3,
    "W7": 3,
    "W8": 4,
    "W9": 0,
    "T1": 3,
    "T2": 4,
    "T3": 4,
    "T4": 3,
    "T5": 2,
    "T6": 3,
    "T7": 3,
    "T8": 3,
    "T9": 2,
    "F1": 1,
    "F2": 0,
    "F3": 1,
    "F4": 1,
    "J1": 2,
    "J2": 3,
    "J3": 2,
}

seatWind = 3
prevailingWind = 0

# hand1["shown"] = pack

i = time.time()
# ret = rule.dist_to_duo_single("W", 7, hand1, tile_list)
# print(ret)
# 输入：手牌，附露，场面上已知的牌墙的信息（去除手牌，所有可见的附露，和打出的牌）， 门风，圈风，搜多少个结果，搜距离， 算番的时候是否禁止胡绝张：false表示算，true表示不算
(
    list1,
    list1id,
    list2,
    list2id,
    list3,
    list3id,
    list4,
    list4id,
) = formMinComb_c(
    hand1, pack, tile_list, seatWind, prevailingWind, 45, 7, 8, True, False
)

list1.append(list1id)
list2.append(list2id)
list3.append(list3id)
list4.append(list4id)

# print(list1, list2, list3, list4)
for i in range(3):
    r, m = find_redundant_and_missing(hand1, list1[i])
    print(r, m, list1[i])

# # print(ret[2])
# # for result in ret[2]:
# #     print(result)
# id = ret[1]
# l = ret[0]
# # print(len(l))
# if id != -1:
#     print(l[id])
# id = ret[3]
# l = ret[2]
# # print(len(l))
# if id != -1:
#     print(l[id])
# id = ret[5]
# l = ret[4]
# # print(len(l))
# if id != -1:
#     print(l[id])
# id = ret[7]
# l = ret[6]
# if id != -1:
#     print(l[id])
# print(time.time() - i)

# i = time.time()
# ret = src_py.generic.form_min_combination(hand, tile_list, 0, 0, 45, 7)
# # )  # ret = formMinComb_c(hand1, pack, tile_list, 0, 0, 45, 7)
# # for l in ret:
# #     print(l[l[-1]])
# # tile_list["T5"] = 0
# # tile_list["W5"] = 0
# # tile_list["B5"] = 0
# for l in ret:
#     print(l[l[-1]])
# print(time.time() - i)
