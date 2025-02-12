# Heuristic AI

import sys
import os

sys.path.append("/data")
import numpy as np
import json
import copy
from collections import defaultdict
from fanCalcLib import formMinComb_c
import rule
import torch
from torch import nn

# import generic


def default_list():
    return []


def default_zero():
    return 0


def default_one():
    return 1


def default_three_zeros():
    return [0, 0, 0]


# Bot wiki: https://wiki.botzone.org.cn/index.php?title=Chinese-Standard-Mahjong#.E6.B8.B8.E6.88.8F.E4.BA.A4.E4.BA.92.E6.96.B9.E5.BC.8F

# NOTE:
# This version from framework loads parameters from model_slim from supervised_learning_v7

# fmt: off
tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
            'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭


]
# fmt: on


class TileProbModuleNew(nn.Module):
    """
    Computation for opponent_desire_prob, held_prob, discard_prob
    """

    def __init__(self):
        super(TileProbModuleNew, self).__init__()

        self.final_layer = nn.Sequential(
            nn.Linear(11, 1),
        )

    def forward(self, meta_feature, tile_wall_feature):
        # input: meta (data[:4])

        meta_feature_expanded = meta_feature.view(-1, 1, 6).repeat(1, 34, 1).view(-1, 6)
        tile_wall_feature_flatten = tile_wall_feature.view(-1, 5)
        concated_feature = torch.cat(
            (meta_feature_expanded, tile_wall_feature_flatten), dim=1
        )
        # print(concated_feature.shape)

        final = self.final_layer(concated_feature)
        return final.view(-1, 34)


def convert_data_representation(data_pre_conversion, key_order_list, default_value):
    """
    convert from dict to one-hot
    """
    data_post_conversion = []
    for key in key_order_list:
        if key in data_pre_conversion.keys():
            data_post_conversion.append(data_pre_conversion[key])
        else:
            data_post_conversion.append(default_value)
    return data_post_conversion


def reversed_tile_conversion(list_rep, key_order_list):
    ret_dict = defaultdict(default_zero)
    for i in range(len(key_order_list)):
        ret_dict[key_order_list[i]] += list_rep[i]
    return ret_dict


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


def construct_network_input(remaining_tile_ct, opponent_held_ct, tile_wall_info):
    """
    construct network input from necessary components
    remaining_tile_count, opponent_held_count -> meta_feature
    tile_wall_info, pack_info, discard_info -> tile_wall_feature
    return meta_feature, tile_wall_feature
    """
    remaining_tile_count, opponent_held_count = (
        remaining_tile_ct + 2,
        opponent_held_ct + 2,
    )
    normalized_rtc, normalized_ohc = (
        remaining_tile_count / 120,
        opponent_held_count / 45,
    )

    meta_feature_new = torch.tensor(
        [
            normalized_rtc,
            1 / remaining_tile_count,
            1 - normalized_rtc,
            normalized_ohc,
            1 / opponent_held_count,
            1 - normalized_ohc,
        ]
    )

    tile_wall_feature_list_raw = [
        tile_wall_info,
    ]
    tile_wall_feature_list = []
    for element in tile_wall_feature_list_raw:
        converted_element = convert_data_representation(element, tile_list_raw, 0)
        slice_feature = decode_tile_wall(converted_element)
        tile_wall_feature_list.append(slice_feature)
    return (
        meta_feature_new.float(),
        torch.tensor(np.stack(tile_wall_feature_list, axis=1)).float(),
    )


def sortation(
    list1,
    list2,
    list3,
    list4,
    already_chi_peng,
    for_QQR=False,
    list2_penalty=0.4,
    list3_penalty=1.5,
    list4_penalty=2.1,
):
    """
    sort results from form_min_comb
    according to 上听数 and list tier, find at least 5default_list
    list1: list 1 from form_min_comb
    list2: list 2 from form_min_comb
    list3: list 3 from form_min_comb
    list4: list 4 from form_min_comb
    list*_penalty: self defined penalty to 上听数, default: list 1 +0, list 2, list 3+1, list 4 + 2
    """
    seen_target_list = []
    dist_penalty = [0, list2_penalty, list3_penalty, list4_penalty]
    dist_dict = defaultdict(default_list)
    list_holder = []
    list_holder.append(list1)
    list_holder.append(list2)
    list_holder.append(list3)
    list_holder.append(list4)
    considered_list_range = 4
    if already_chi_peng:
        considered_list_range = 2
    if for_QQR:
        considered_list_range = 4
        dist_penalty = [0, 0, 0, 0]
    for i in range(considered_list_range):
        if len(list_holder[i]) > 1:
            for entry in list_holder[i]:
                if (isinstance(entry, list) or isinstance(entry, tuple)) and len(
                    entry
                ) == 6:
                    entry = list(entry)
                    dist = entry[0] + dist_penalty[i]
                    # append indicator for list tier at the end of entry list
                    entry.append(i)
                    if i < 2:
                        original_set = entry[5]
                        # 不算吃碰相关番种
                        list_of_fan = {
                            "不求人",
                            "门前清",
                            "暗杠",
                            "双暗刻",
                            "双暗杠",
                            "三暗刻",
                            "四暗刻",
                        }
                        valid_fan_set = original_set - list_of_fan
                        entry[5] = valid_fan_set
                    if i < 1:
                        original_set = entry[5]
                        # 不算不求人番种
                        list_of_fan = {"边张", "嵌张", "单钓将"}
                        valid_fan_set = original_set - list_of_fan
                        entry[5] = valid_fan_set
                    if for_QQR:
                        # 增加全求人番种
                        original_set = entry[5]
                        original_set.add("全求人")
                        # 删除不求人相关番种，及不合格番种
                        illegal_fan = {
                            "门前清",
                            "不求人",
                            "单钓将",
                            "双暗刻",
                            "三暗刻",
                            "四暗刻",
                            "暗杠",
                            "双暗杠",
                            "无番和",
                        }
                        impossible_target_list = {
                            "九莲宝灯",
                            "连七对",
                            "十三幺",
                            "七对",
                            "全不靠",
                            "七星不靠",
                        }
                        if len(impossible_target_list - original_set) != 6:
                            continue
                        valid_fan_set = original_set - illegal_fan
                        entry[5] = valid_fan_set
                    dist_dict[dist].append(entry)
    result_list = []
    # dict_dist_list = []
    for e in range(100):
        i = e / 10
        len_dict = len(dist_dict[i])
        # dict_dist_list.extend([i for _ in range(len_dict)])
        for result_case in dist_dict[i]:
            target_hash = rule.hash_seperated_custom_tile(result_case[4])
            if target_hash not in seen_target_list:
                seen_target_list.append(target_hash)
                result_list.append(result_case)

    return (
        result_list  # ,  min(dict_dist_list), sum(dict_dist_list)/len(dict_dist_list)
    )


def find_missing_tile(hand_list, selected_tiles):
    missing_tile = []
    for t in hand_list:
        if t not in selected_tiles:
            selected_tiles[t] = 0
        if hand_list[t] - selected_tiles[t] > 0:
            missing_tile.append(t)
    return missing_tile


def bz_chi_decode(tile):
    """
    Decode to dictionary style {X:1, Y:1, Z:1}
    """
    kind = tile[0]
    rank = int(tile[1])
    return {kind + str(rank - 1): 1, tile: 1, kind + str(rank + 1): 1}


def bz_peng_decode(prev_played_tile):
    """
    Decode to dictionary style {X:3}
    """
    return {prev_played_tile: 3}


def calculate_target_probability(
    list_element,
    tile_list,
    remaining_tileWall=50,
    opponent_hidden_tiles=39,
    held_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_zero),
    chi_peng_count_remain=0,
):
    """
    calculate win probability towards "target"
    list_element: entries produced by sortation, after formMinComb_c
    held_prob: probability of tile held by other player, probability is assigned to each tile individually, default to 0
                某张牌被别人拿住的概率, 默认为0
    Each tile X's acquisition probability equals:
    list0,1 (self-draw): (X's remaining count / total remaining tiles) * (1 - held_prob[x])
    list0,1 (chi, peng): X's remaining count * held_prob[x] * discard_prob[x]
    list2,3 (self-draw): (X's remaining count / total remaining tiles) * (1 - held_prob[x])
    Merged_list probability: intersection probability within entry, union probability between entries
    return: a probability, a list of desired tiles, a indicator of if chi/peng is allowed
    """

    list_probability = 100
    target_fan_list = list_element[5]
    accumulated_fan_preference = 0
    for target_fan in target_fan_list:
        if target_fan == "全不靠":
            accumulated_fan_preference += fan_preference[target_fan] * 21.0
        else:
            accumulated_fan_preference += fan_preference[target_fan]
    is_QQR = False
    if "全求人" in target_fan_list:
        is_QQR = True

    effective_chi_peng_count_remain = 0
    if is_QQR:
        effective_chi_peng_count_remain = chi_peng_count_remain

    # find tiles in need, number in need
    missing_tile = {
        "CHI": defaultdict(default_zero),
        "PENG": defaultdict(default_zero),
        "OTHER": defaultdict(default_zero),
    }  # 管理当前 list
    # TODO: missing_tile, desired_tile_list calculate seperately, no intervention
    targ = copy.deepcopy(list_element[4])
    sel = copy.deepcopy(list_element[1])
    # 将，无法吃碰，吃，上家，碰，3家
    # 要求，只有3张牌中一张可以吃碰
    for ele in targ:
        if "AnGang" in ele:
            del ele["AnGang"]
        ele_kind = "DUO"
        for k in ele:
            # encode k for trio, duo, and straight
            if len(ele) != 1:
                # straight
                ele_kind = "STRAIGHT"
            elif ele[k] == 2:
                # duo
                ele_kind = "DUO"
            else:
                # trio
                ele_kind = "TRIO"
            if k in sel:
                valid_count = min(ele[k], sel[k])
                ele[k] -= valid_count
                sel[k] -= valid_count
                if sel[k] == 0:
                    del sel[k]

        for k in ele:
            if ele[k] != 0:
                if sum(ele.values()) == 1:
                    if ele_kind == "TRIO":
                        missing_tile["PENG"][k] += ele[k]
                    elif ele_kind == "STRAIGHT":
                        missing_tile["CHI"][k] += ele[k]
                    else:
                        missing_tile["OTHER"][k] += ele[k]
                else:
                    missing_tile["OTHER"][k] += ele[k]

    # combine same tile
    tile_dict = defaultdict(default_three_zeros)
    for dict_key in ["CHI", "PENG", "OTHER"]:
        for k in missing_tile[dict_key]:
            k_count_need = int(missing_tile[dict_key][k])
            tile_dict[k][0] += k_count_need
            if dict_key == "PENG":
                tile_dict[k][1] = 1
                tile_dict[k][2] = 1
            if dict_key == "CHI":
                tile_dict[k][2] = 1

    # if this target allows for chi_peng, 1: disallow chi/peng, 0: allow chi/peng
    total_tiles_needed = 0
    for k in tile_dict:
        total_k_need, can_peng, able_chi_peng = tile_dict[k]
        total_tiles_needed += total_k_need
        self_draw_prob = 1

        if tile_list[k] < total_k_need:
            list_probability *= 0
            break

        # self draw probability
        for i in range(total_k_need):
            self_draw_prob *= (tile_list[k] - i) / (remaining_tileWall - i)
        self_draw_prob *= held_prob[k] ** total_k_need

        # chi-peng probability
        chi_peng_prob = 0

        if list_element[-1] in [0, 1]:
            # list 0 and 1 are allowed to chi and peng
            chi_peng_prob = tile_list[k] / opponent_hidden_tiles
            if can_peng != 0:
                # peng: from 3 opponents
                chi_peng_prob *= 2
            if able_chi_peng == 0:
                # duo: not allowed to chi or peng
                chi_peng_prob *= 0
        list_probability *= chi_peng_prob + self_draw_prob
    # apply importance manipulation by fan
    list_probability *= accumulated_fan_preference
    return list_probability


def perform_target_search(
    hand_list_cp, pack, tile_list, seat_wind, prevailing_wind, disable_juezhang
):
    # 8-fan targets
    (list1, list1id, list2, list2id, list3, list3id, list4, list4id) = formMinComb_c(
        hand_list_cp,
        pack,
        tile_list,
        seat_wind,
        prevailing_wind,
        80,
        8,
        8,
        True,
        disable_juezhang,
    )
    already_chi_peng = False
    for ent in pack:
        if "AnGang" not in ent.keys():
            already_chi_peng = True

    list1.append(list1id)
    list2.append(list2id)
    list3.append(list3id)
    list4.append(list4id)
    sorted_list = sortation(list1, list2, list3, list4, already_chi_peng)
    if len(sorted_list) > 64:
        segmented_list_8_fan = sorted_list[:64]
    else:
        segmented_list_8_fan = sorted_list
    # # 2-fan targets for 不求人
    # (list1, list1id, list2, list2id, list3, list3id, list4, list4id) = formMinComb_c(
    #     hand_list_cp,
    #     pack,
    #     tile_list,
    #     seat_wind,
    #     prevailing_wind,
    #     8,
    #     6,
    #     2,
    #     False,
    #     disable_juezhang,
    # )
    # list1.append(list1id)
    # list2.append(list2id)
    # list3.append(list3id)
    # list4.append(list4id)
    # sorted_list = sortation(list1, list2, list3, list4, already_chi_peng, for_QQR=True)
    # if len(sorted_list) > 8:
    #     segmented_list_2_fan = sorted_list[:8]
    # else:
    #     segmented_list_2_fan = sorted_list
    return segmented_list_8_fan  # + segmented_list_2_fan


def ejection(
    hand_list,
    tile_list,
    seat_wind,
    prevailing_wind,
    disable_juezhang=False,
    held_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    tile_preference=defaultdict(default_one),
    opponent_hidden_tiles=39,
):
    """
    select the tile with least win-probability and eject
    Implementation: compute the sum of win-prob when ejecting TILE, and select the tile with the max win-prob
    return: action, tile, max_prob
    when called by itself, defaults to self-draw
    otherwise, defaults to chi/peng considerations
    """
    # print(hand_list)
    # print(tile_list)
    # print(seat_wind, prevailing_wind)

    ejection_candidates = defaultdict(default_zero)

    remaining_tileWall = 0
    for k in tile_list:
        remaining_tileWall += tile_list[k]

    held_prob_cp = held_prob.copy()
    # calculate held_prob if default
    if len(held_prob) == 0:
        for k in tile_list:
            held_prob_cp[k] = 1.0 * opponent_hidden_tiles / remaining_tileWall

    hand_list_cp = copy.deepcopy(hand_list)
    pack = hand_list_cp["shown"].copy()

    chi_peng_count_remain = 4
    for ent in pack:
        if "AnGang" not in ent.keys():
            chi_peng_count_remain -= 1

    del hand_list_cp["shown"]
    segmented_list = perform_target_search(
        hand_list_cp, pack, tile_list, seat_wind, prevailing_wind, disable_juezhang
    )
    for list_element in segmented_list:
        win_prob = calculate_target_probability(
            list_element,
            tile_list,
            remaining_tileWall,
            opponent_hidden_tiles,
            held_prob_cp,
            fan_preference,
            chi_peng_count_remain,
        )
        # print("{:.3e}: {} {}".format(win_prob, list_element[-2], list_element[-3]))
        missing_tile_list = find_missing_tile(hand_list_cp, list_element[1])
        # handle special case, in which a new win-tile is required for hu (i.e. 单钓将)
        if (
            (len(missing_tile_list) == 0 or win_prob == 100)
            and list_element[-1] % 2 == 1
            and len(list_element[3]) != 0
        ):
            last_tile_selection_list = list(list_element[3])
            for t in last_tile_selection_list:
                ejection_candidates[t] += 100 * (
                    tile_list[t] / remaining_tileWall * (1 - held_prob_cp[t])
                    + tile_list[t] * held_prob_cp[t] * 3
                )
            # print(ejection_candidates)
        for t in missing_tile_list:
            ejection_candidates[t] += win_prob

    effective_hand_tile_list = []
    for k in hand_list_cp:
        if hand_list_cp[k] > 0:
            effective_hand_tile_list.append(k)
    for t in effective_hand_tile_list:
        ejection_candidates[t] = ejection_candidates[t] * tile_preference[t]

    # print(ejection_candidates)

    # handle possible BuGang or AnGang
    max_ejection_candidate, max_prob = list(
        sorted(ejection_candidates.items(), key=lambda item: -item[1])[0]
    )
    if tile_list[max_ejection_candidate] == 0:
        if hand_list[max_ejection_candidate] == 4:
            return ("GANG", max_ejection_candidate, max_prob)
        for entry in pack:
            if entry.get(max_ejection_candidate, 0) == 3:
                return ("BUGANG", max_ejection_candidate, max_prob)
    return ("PLAY", max_ejection_candidate, max_prob)


def calc_win_prob(
    hand_list,
    tile_list,
    seat_wind,
    prevailing_wind,
    disable_juezhang=False,
    held_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    opponent_hidden_tiles=39,
):
    """
    calculate win-prob WHEN tile-count == 13
    return: current win prob
    """
    current_win_prob = 0

    remaining_tileWall = 0
    for k in tile_list:
        remaining_tileWall += tile_list[k]

    held_prob_cp = held_prob.copy()
    # calculate held_prob if default
    if len(held_prob) == 0:
        for k in tile_list:
            held_prob_cp[k] = 1.0 * opponent_hidden_tiles / remaining_tileWall

    hand_list_cp = copy.deepcopy(hand_list)
    pack = hand_list_cp["shown"].copy()

    chi_peng_count_remain = 4
    for ent in pack:
        if "AnGang" not in ent.keys():
            chi_peng_count_remain -= 1

    del hand_list_cp["shown"]
    segmented_list = perform_target_search(
        hand_list_cp, pack, tile_list, seat_wind, prevailing_wind, disable_juezhang
    )
    for list_element in segmented_list:
        win_prob = calculate_target_probability(
            list_element,
            tile_list,
            remaining_tileWall,
            opponent_hidden_tiles,
            held_prob_cp,
            fan_preference,
            chi_peng_count_remain,
        )
        current_win_prob += win_prob
    return current_win_prob


def chi_peng_utility(new_tile, test_chi):
    """
    return possible combinations of chi/peng, (action, midtile(for chi), tile-combination)
    """
    new_tile_type = new_tile[0]
    new_tile_rank = int(new_tile[1])
    test_case_list = []
    if test_chi and new_tile_type in ["B", "W", "T"]:
        for low_rank in range(max(1, new_tile_rank - 2), min(8, new_tile_rank + 1)):
            test_case = {
                new_tile_type + str(low_rank): 1,
                new_tile_type + str(low_rank + 1): 1,
                new_tile_type + str(low_rank + 2): 1,
            }

            test_case_list.append(["CHI", new_tile_type + str(low_rank + 1), test_case])
    test_case_list.append(["PENG", new_tile, {new_tile: 3}])
    return test_case_list


def test_chi_peng(
    hand_list,
    tile_list,
    new_tile,
    seat_wind,
    prevailing_wind,
    test_chi,
    held_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    tile_preference=defaultdict(default_one),
    opponent_hidden_tiles=39,
):
    """
    tentatively form chi\peng, then calculate win-prob, if win-prob is meaningfully larger, then perform the action
    gang is considered as a special case in peng
    """
    prev_hand_list = hand_list.copy()
    prev_hand_list[new_tile] -= 1
    prev_max_prob = calc_win_prob(
        prev_hand_list,
        tile_list,
        seat_wind,
        prevailing_wind,
        False,
        held_prob,
        fan_preference,
        opponent_hidden_tiles,
    )
    test_case_list = chi_peng_utility(new_tile, test_chi)
    test_case_result = []
    for action, mid_tile, test_case in test_case_list:
        # Test if chi/peng pre-requisite is satisfied
        prerequisite_satisfied = True
        for t in test_case:
            if hand_list.get(t, 0) < test_case[t]:
                prerequisite_satisfied = False

        if prerequisite_satisfied:
            hand_list_cp = copy.deepcopy(hand_list)
            for t in test_case:
                hand_list_cp[t] -= test_case[t]
            hand_list_cp["shown"].append(test_case)
            (eject_action, eject_tile, max_prob) = ejection(
                hand_list_cp,
                tile_list,
                seat_wind,
                prevailing_wind,
                True,
                held_prob,
                fan_preference,
                tile_preference,
                opponent_hidden_tiles,
            )
            test_case_result.append(
                (
                    action,
                    mid_tile,
                    eject_action,
                    eject_tile,
                    max_prob,
                )
            )

    if len(test_case_result) == 0:
        return "PASS", None, None, 0
    best_action_case = sorted(test_case_result, key=lambda item: -item[4])[0]
    (action, mid_tile, eject_action, eject_tile, max_prob) = best_action_case
    if best_action_case[4] < prev_max_prob:
        return "PASS", None, None, 0
    else:
        if action == "PENG" and new_tile == eject_tile and eject_action == "BUGANG":
            return "GANG", new_tile, None, max_prob
        else:
            return action, mid_tile, eject_tile, max_prob


if __name__ == "__main__":
    # tile_list
    tile_list = {}
    prev_played_tile = None
    last_action_is_draw_flag = False
    opponents_hidden_tile_count = 39
    for tile in tile_list_raw:
        tile_list[tile] = 4

    # collect discard history for
    discard_history_dict_list = [
        defaultdict(default_zero) for _ in range(3)
    ]  # discard history for other 3 players
    pack_history_dict_list = [
        defaultdict(default_zero) for _ in range(3)
    ]  # pack history for other 3 players

    # load params
    path_to_params = "/data/extracted_model_slim_params"
    # path_to_params = "extracted_model_slim_params/top2"
    prob_module = TileProbModuleNew()
    prob_module.load_state_dict(
        torch.load(
            "{}/net.pkl".format(path_to_params), map_location=torch.device("cpu")
        )
    )
    with open("{}/global.json".format(path_to_params), "r") as f:
        tmp = json.load(f)
        fan_param_dict = tmp["fan"]
        tile_param_dict = tmp["tile"]

    fan_param_dict["五门齐"] *= 10

    input()  # 1
    while True:
        request = input()
        remaining_tileWall = 0
        for k in tile_list:
            remaining_tileWall += tile_list[k]
        while not request.strip():
            request = input()
        t = request.split()
        if t[0] == "0":
            # initial information
            seatWind = int(t[1])
            prevailing_wind = int(t[2])
            print("PASS")
        elif t[0] == "1":
            # dealing hand
            canonical_hand = t[5:]
            hand_list = rule.from_canonical_to_custom_encoding(canonical_hand)
            hand_list["shown"] = []
            tile_list = rule.update_tile_info(tile_list, hand_list)
            print("PASS")
        elif t[0] == "2":
            new_tile = t[1]
            hand_list[new_tile] += 1
            tile_list = rule.update_tile_info(tile_list, {new_tile: 1})

            if rule.check_hu(
                hand_list, tile_list, new_tile, seatWind, prevailing_wind, True
            ):
                # check hu
                print("HU")
            else:
                # tile draw
                remaining_tileWall -= 1
                meta_feature, tile_wall_feature = construct_network_input(
                    remaining_tileWall,
                    opponents_hidden_tile_count,
                    tile_list,
                )
                hp = (
                    prob_module(meta_feature, tile_wall_feature)
                    .data.numpy()
                    .tolist()[0]
                )
                hp_dict = reversed_tile_conversion(hp, tile_list_raw)
                (action, eject_tile, max_prob) = ejection(
                    hand_list,
                    tile_list,
                    seatWind,
                    prevailing_wind,
                    held_prob=hp_dict,
                    fan_preference=fan_param_dict,
                    tile_preference=tile_param_dict,
                    opponent_hidden_tiles=opponents_hidden_tile_count,
                )
                # check gang, check bugang, otherwise play
                # print(action, tile)
                cmd = "{} {}".format(action, eject_tile)
                if action == "PLAY":
                    prev_played_tile = eject_tile
                if action == "GANG":
                    angang = eject_tile
                print(cmd)
        elif t[0] == "3":
            p = int(t[1])
            if t[2] == "DRAW":
                last_action_is_draw_flag = True
                print("PASS")
            elif t[2] == "GANG":
                if p == seatWind and angang:
                    # angang, self
                    hand_list[angang] = 0
                    del hand_list[angang]
                    hand_list["shown"].append({angang: 4, "AnGang": True})
                    print("PASS")
                elif last_action_is_draw_flag:
                    # angang, other player
                    opponents_hidden_tile_count += 1
                    print("PASS")
                else:
                    opponents_hidden_tile_count -= 3
                    # gang
                    tile_list = rule.update_tile_info(tile_list, {prev_played_tile: 3})

                    # update pack history
                    pack_history_offset = (3 + p - seatWind) % 4
                    pack_history_dict_list[pack_history_offset][prev_played_tile] = 4
                print("PASS")

            elif t[2] == "BUGANG":
                if p == seatWind:
                    # bugang, self
                    hand_list[t[3]] = 0
                    del hand_list[t[3]]
                    hand_list["shown"].append({t[3]: 4})
                    print("PASS")
                else:
                    # update pack history
                    pack_history_offset = (3 + p - seatWind) % 4
                    pack_history_dict_list[pack_history_offset][t[3]] += 1

                    tile_list = rule.update_tile_info(tile_list, {t[3]: 1})
                    hand_list_cp = copy.deepcopy(hand_list)
                    hand_list_cp[t[3]] += 1
                    if rule.check_hu(
                        hand_list_cp,
                        tile_list,
                        prev_played_tile,
                        seatWind,
                        prevailing_wind,
                        False,
                        True,
                    ):
                        print("HU")
                    else:
                        print("PASS")
            else:
                angang = None
                last_action_is_draw_flag = False
                ## Need to work on chi/peng action
                if t[2] == "CHI":
                    observed_tile_dict = bz_chi_decode(t[3])
                    observed_tile_dict_adj = observed_tile_dict.copy()
                    try:
                        # adjust for previously played tile, remove it from dict, prevent double counting
                        # 之前被吃的牌已经算过一次了，从dict中除去，防止双重计数
                        observed_tile_dict_adj[prev_played_tile] -= 1
                    except Exception:
                        print(
                            "ERROR in adjusting bz_chi_decode, mid_tile: {}, prev_played_tile: {}".format(
                                t[3], prev_played_tile
                            )
                        )
                    if p == seatWind:
                        # self, update pack info
                        for ob_tile in observed_tile_dict_adj:
                            hand_list[ob_tile] -= observed_tile_dict_adj[ob_tile]
                        hand_list["shown"].append(observed_tile_dict)
                    else:
                        # update pack history
                        pack_history_offset = (3 + p - seatWind) % 4
                        for tile_t in observed_tile_dict:
                            pack_history_dict_list[pack_history_offset][tile_t] += 3

                        # update hidden tile count
                        opponents_hidden_tile_count -= 2
                        tile_list = rule.update_tile_info(
                            tile_list, observed_tile_dict_adj
                        )
                elif t[2] == "PENG":
                    observed_tile_dict = bz_peng_decode(prev_played_tile)
                    observed_tile_dict_adj = observed_tile_dict.copy()
                    observed_tile_dict_adj[prev_played_tile] -= 1

                    if p == seatWind:
                        hand_list[prev_played_tile] -= observed_tile_dict_adj[
                            prev_played_tile
                        ]
                        hand_list["shown"].append(observed_tile_dict)
                    else:
                        # update pack history
                        pack_history_offset = (3 + p - seatWind) % 4
                        for tile_t in observed_tile_dict:
                            pack_history_dict_list[pack_history_offset][tile_t] += 3

                        opponents_hidden_tile_count -= 2
                        tile_list = rule.update_tile_info(
                            tile_list, observed_tile_dict_adj
                        )

                # played tile: t[-1]
                prev_played_tile = t[-1]
                if p == seatWind:
                    # tile_list already updated
                    hand_list[t[-1]] -= 1
                    print("PASS")
                else:
                    # update discard history
                    # update tile_list
                    # check hu

                    # update discard history
                    discard_history_offset = (3 + p - seatWind) % 4
                    discard_history_dict_list[discard_history_offset][
                        prev_played_tile
                    ] += 1

                    # update tile_list
                    hand_list_cp = copy.deepcopy(hand_list)
                    hand_list_cp[t[-1]] += 1
                    tile_list = rule.update_tile_info(tile_list, {t[-1]: 1})

                    # check hu
                    if rule.check_hu(
                        hand_list_cp,
                        tile_list,
                        prev_played_tile,
                        seatWind,
                        prevailing_wind,
                        False,
                        False,
                    ):
                        print("HU")

                    # check if test chi (only happen if last player is 上家)
                    if p == (seatWind + 3) % 4:
                        test_chi = True
                    else:
                        test_chi = False

                    meta_feature, tile_wall_feature = construct_network_input(
                        remaining_tileWall,
                        opponents_hidden_tile_count,
                        tile_list,
                    )
                    hp = (
                        prob_module(meta_feature, tile_wall_feature)
                        .data.numpy()
                        .tolist()[0]
                    )
                    hp_dict = reversed_tile_conversion(hp, tile_list_raw)
                    (action, mid_tile, eject_tile, max_prob) = test_chi_peng(
                        hand_list_cp,
                        tile_list,
                        t[-1],
                        seatWind,
                        prevailing_wind,
                        test_chi,
                        held_prob=hp_dict,
                        fan_preference=fan_param_dict,
                        tile_preference=tile_param_dict,
                        opponent_hidden_tiles=opponents_hidden_tile_count,
                    )

                    if action == "PENG":
                        print("PENG {}".format(eject_tile))
                    elif action == "CHI":
                        print("CHI {} {}".format(mid_tile, eject_tile))
                    elif action == "GANG":
                        print("GANG")

                    print("PASS")
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()
