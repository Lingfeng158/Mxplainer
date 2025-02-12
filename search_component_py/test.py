import numpy as np
import feature
import rule
import generic
from collections import defaultdict
import time

hand_list = {"B5": 3, "B2": 1, "W8": 1, "B1": 1, "B4": 1, "W9": 1, "W7": 1, "B6": 1}
pack = [{"T1": 1, "T2": 1, "T3": 1}]
tile_wall = {
    "B1": 1,
    "B2": 2,
    "B3": 3,
    "B4": 2,
    "B5": 0,
    "B6": 2,
    "B7": 1,
    "B8": 4,
    "B9": 4,
    "W1": 2,
    "W2": 3,
    "W3": 4,
    "W4": 3,
    "W5": 1,
    "W6": 3,
    "W7": 2,
    "W8": 3,
    "W9": 2,
    "T1": 2,
    "T2": 1,
    "T3": 2,
    "T4": 2,
    "T5": 2,
    "T6": 0,
    "T7": 1,
    "T8": 0,
    "T9": 3,
    "F1": 3,
    "F2": 1,
    "F3": 1,
    "F4": 2,
    "J1": 3,
    "J2": 0,
    "J3": 2,
}
hand_list["shown"] = pack
seat = 0
pWind = 3
lookup_table = {}
# lookup_table["B" + "T0D0"] = [(0, {}, 0, [], {}, {})]
ret = generic.one_type_lookup_dp("W", hand_list, tile_wall)
# ret = generic.one_type_all_combination_dp("B", hand_list, tile_wall, 1, 0, lookup_table)
# ret = generic.one_type_all_combination_dp("B", hand_list, tile_wall, 2, 0, lookup_table)
# ent_list = ret["BT1D1"]
for key in ret:
    print(key, len(ret[key]))
# for ent in ent_list:
#     print(
#         ent[0],
#         rule.hash_custom_tiles(ent[1]),
#         rule.hash_seperated_custom_tile(ent[3]),
#         rule.hash_custom_tiles(ent[4]),
#         rule.hash_custom_tiles(ent[5]),
#     )
# i_t = time.time()
# # for i in range(100):
ret = generic.form_min_combination(
    hand_list,
    tile_wall,
    seat,
    pWind,
    result_threshold=32,
    max_dist=5,
    target_fan_val=8,
)
for ent_list in ret:
    print(len(ent_list))
# print(time.time() - i_t)

# min_dist = 9
# for r in ret:

#     for l in r:
#         if isinstance(l, list):
#             min_dist = min(min_dist, l[0])
# print(min_dist)


# i_t = time.time()
# for i in range(100):
#     ret = generic.one_type_lookup_dp("B", hand_list, tile_wall)
# # # for key in ret.keys():
# # #     print(key, len(ret[key]))
# # # print(ret["BT3D1"][:3])

# # ct = 0
# # for k in ret.keys():
# #     dist_dict = defaultdict(int)
# #     for ent in ret[k]:
# #         dist_dict[ent[0]] += 1
# #     print(k, dist_dict)
# # print(dist_dict)
# print(time.time() - i_t)

# i_t = time.time()
# for i in range(100):
#     ret = generic.one_type_lookup("B", hand_list, tile_wall)
# print(time.time() - i_t)
# # # # for key in ret.keys():
# # # #     print(key, len(ret[key]))
# # # # print(ret["BT3D1"][:3])
# # dist_dict = defaultdict(int)
# # for ent in ret["TT1D1"]:
# #     dist_dict[ent[0]] += 1
# #     if ent[0] == 1:
# #         print(ent)
# #         break
# # print(dist_dict)
# # # print(ret)
# # ret = generic.one_type_all_combination_alt("B", hand_list, tile_wall, 2, 1, 7)
