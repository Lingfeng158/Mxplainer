# Heuristic AI

import sys
import os

sys.path.append("/data")
import numpy as np
import json
import copy
import random
from collections import defaultdict
import time
from fanCalcLib import formMinComb_c
import rule

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


# Programming goal:
# tile ejection: select least probable tile from heuristic calculation
# win-probability after tile ejection: win-probability = probability of max probable tile from heuristic calculation
# chi/peng: check 3 posibilities of chi and 1 posibility of peng, then eject one tile, compare new win-prob against prev-win-prob
# gang: if a tile can be discarded, and a trio is in hand, form anGang, a trio pack is in hand, form buGang

# TODO: iteratively improve logic


# fmt: off
tile_list_raw = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',  #饼
            'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',   #万
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9',   #条
            'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3' #风、箭
]
# fmt: on

_tile_pref = {'B1':0.99, 'B2':0.991, 'B3':0.992, 'B4':0.993, 'B5':0.994, 'B6':0.995, 'B7':0.996, 'B8':0.996, 'B9':0.997,  #饼
            'W1':0.998, 'W2':0.999, 'W3':1.0, 'W4':1.001, 'W5':1.002, 'W6':1.003, 'W7':1.004, 'W8':1.005, 'W9':1.006,   #万
            'T1':0.989, 'T2':0.988, 'T3':0.987, 'T4':0.987, 'T5':0.986, 'T6':0.985, 'T7':0.984, 'T8':0.983, 'T9':0.982,   #条
            'F1':1.007, 'F2':1.008, 'F3':1.009, 'F4':1.01, 'J1':1.011, 'J2':1.012, 'J3':1.013 #风、箭
}


def sortation(
    list1,
    list2,
    list3,
    list4,
    already_chi_peng,
    list2_penalty=0,
    list3_penalty=0,
    list4_penalty=1,
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
    for i in range(considered_list_range):
        if len(list_holder[i]) > 1:
            for entry in list_holder[i]:
                if (isinstance(entry, list) or isinstance(entry, tuple)) and len(
                    entry
                ) == 6:
                    if "七对" in entry[-1] or "连七对" in entry[-1]:
                        entry = list(entry)
                        dist = entry[0] + dist_penalty[i]
                        # append indicator for list tier at the end of entry list
                        entry.append(i)
                        dist_dict[dist].append(entry)
    result_list = []
    for i in range(8):
        for result_case in dist_dict[i]:
            target_hash = rule.hash_seperated_custom_tile(result_case[4])
            if target_hash not in seen_target_list:
                seen_target_list.append(target_hash)
                result_list.append(result_case)

    return result_list


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
    held_prob=defaultdict(default_zero),
    discard_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    straight_coeff=1,
    trio_coeff=1,
):
    """
    calculate win probability towards "target"
    list_element: entries produced by sortation, after formMinComb_c
    held_prob: probability of tile held by other player, probability is assigned to each tile individually, default to 0
                某张牌被别人拿住的概率, 默认为0
    discard_prob: probability of tile held by other players, but other players play the tile, default to 0
                某张牌被别人拿住, 但之后打出的概率
    Each tile X's acquisition probability equals:
    list0,1 (self-draw): (X's remaining count / total remaining tiles) * (1 - held_prob[x])
    list0,1 (chi, peng): X's remaining count * held_prob[x] * discard_prob[x]
    list2,3 (self-draw): (X's remaining count / total remaining tiles) * (1 - held_prob[x])
    Merged_list probability: intersection probability within entry, union probability between entries
    return: a probability, a list of desired tiles, a indicator of if chi/peng is allowed
    """

    list_probability = 100
    target_fan_list = list_element[5]
    for target_fan in target_fan_list:
        list_probability *= fan_preference[target_fan]
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

        # control the preference for trio/straight
        if ele_kind == "TRIO":
            list_probability *= trio_coeff
        if ele_kind == "STRAIGHT":
            list_probability *= straight_coeff

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
                tile_dict[k][1] += k_count_need
            if dict_key == "CHI":
                tile_dict[k][2] += k_count_need

    for k in tile_dict:
        total_k_need, peng_k, chi_k = tile_dict[k]
        self_draw_prob = 1

        if tile_list[k] < total_k_need:
            list_probability *= 0
            break

        for i in range(total_k_need):
            self_draw_prob *= (
                (tile_list[k] - i) / (remaining_tileWall - i) * (1 - held_prob[k])
            )
        chi_peng_prob = 0
        if list_element[-1] in [0, 1]:
            # list 0 and 1 are allowed to chi and peng
            chi_peng_prob = tile_list[k] * held_prob[k] * discard_prob[k]
            if peng_k != 0:
                # peng: from 3 opponents
                chi_peng_prob *= 3
            if peng_k + chi_k == 0:
                # duo: not allowed to chi or peng
                chi_peng_prob *= 0
        list_probability *= chi_peng_prob + self_draw_prob
    return list_probability


def ejection(
    hand_list,
    tile_list,
    seat_wind,
    prevailing_wind,
    disable_juezhang=False,
    held_prob=defaultdict(default_zero),
    discard_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    tile_preference=defaultdict(default_one),
    opponent_desire_prob=defaultdict(default_zero),
    opponent_hidden_tiles=39,
    straight_coeff=1,
    trio_coeff=1,
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
            held_prob_cp[k] = 1.0 * tile_list[k] / remaining_tileWall

    discard_prob_cp = discard_prob.copy()
    # calculate discard_prob if default
    if len(discard_prob) == 0:
        for k in tile_list:
            discard_prob_cp[k] = 1 / opponent_hidden_tiles

    hand_list_cp = copy.deepcopy(hand_list)
    pack = hand_list_cp["shown"].copy()

    already_chi_peng = False
    for ent in pack:
        if 'AnGang' not in ent.keys():
            already_chi_peng = True

    del hand_list_cp["shown"]
    (list1, list1id, list2, list2id, list3, list3id, list4, list4id) = formMinComb_c(
        hand_list_cp,
        pack,
        tile_list,
        seat_wind,
        prevailing_wind,
        32,
        7,
        8,
        True,
        disable_juezhang,
    )

    list1.append(list1id)
    list2.append(list2id)
    list3.append(list3id)
    list4.append(list4id)
    sorted_list = sortation(list1, list2, list3, list4, already_chi_peng)
    if len(sorted_list) > 32:
        segmented_list = sorted_list[:32]
    else:
        segmented_list = sorted_list
    for list_element in segmented_list:
        win_prob = calculate_target_probability(
            list_element,
            tile_list,
            remaining_tileWall,
            held_prob_cp,
            discard_prob_cp,
            fan_preference,
            straight_coeff,
            trio_coeff,
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
                    + tile_list[t] * held_prob_cp[t] * discard_prob_cp[t] * 3
                )
            # print(ejection_candidates)
        for t in missing_tile_list:
            ejection_candidates[t] += win_prob

    effective_hand_tile_list = []
    for k in hand_list_cp:
        if hand_list_cp[k]>0:
            effective_hand_tile_list.append(k)
    for t in effective_hand_tile_list:
        ejection_candidates[t] = (
            ejection_candidates[t] * tile_preference[t] - opponent_desire_prob[t]
        )

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
    discard_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    opponent_hidden_tiles=39,
    straight_coeff=1,
    trio_coeff=1,
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

    discard_prob_cp = discard_prob.copy()
    # calculate discard_prob if default
    if len(discard_prob) == 0:
        for k in tile_list:
            discard_prob_cp[k] = 1 / opponent_hidden_tiles

    hand_list_cp = copy.deepcopy(hand_list)
    pack = hand_list_cp["shown"].copy()

    already_chi_peng = False
    for ent in pack:
        if 'AnGang' not in ent.keys():
            already_chi_peng = True

    del hand_list_cp["shown"]
    (list1, list1id, list2, list2id, list3, list3id, list4, list4id) = formMinComb_c(
        hand_list_cp,
        pack,
        tile_list,
        seat_wind,
        prevailing_wind,
        32,
        7,
        8,
        True,
        disable_juezhang,
    )

    list1.append(list1id)
    list2.append(list2id)
    list3.append(list3id)
    list4.append(list4id)
    sorted_list = sortation(list1, list2, list3, list4, already_chi_peng)
    if len(sorted_list) > 32:
        segmented_list = sorted_list[:32]
    else:
        segmented_list = sorted_list
    for list_element in segmented_list:
        win_prob = calculate_target_probability(
            list_element,
            tile_list,
            remaining_tileWall,
            held_prob_cp,
            discard_prob_cp,
            fan_preference,
            straight_coeff,
            trio_coeff,
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
    play_efficiency_lb=1.5,
    held_prob=defaultdict(default_zero),
    discard_prob=defaultdict(default_zero),
    fan_preference=defaultdict(default_one),
    tile_preference=defaultdict(default_one),
    opponent_desire_prob=defaultdict(default_zero),
    opponent_hidden_tiles=39,
    straight_coeff=1,
    trio_coeff=1,
    chi_coeff=1,
    peng_coeff=1,
):
    """
    tentatively form chi\peng, then calculate win-prob, if win-prob is meaningfully larger, then perform the action
    gang is considered as a special case in peng
    """
    return "PASS", None, None, 0
    

if __name__ == "__main__":
    # tile_list
    tile_list = {}
    prev_played_tile = None
    last_action_is_draw_flag = False
    opponents_hidden_tile_count = 39
    for tile in tile_list_raw:
        tile_list[tile] = 4

    input()  # 1
    while True:
        request = input()
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
                (action, eject_tile, max_prob) = ejection(
                    hand_list,
                    tile_list,
                    seatWind,
                    prevailing_wind,
                    held_prob=defaultdict(default_zero),
                    discard_prob=defaultdict(default_zero),
                    tile_preference=_tile_pref,
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
                print("PASS")

            elif t[2] == "BUGANG":
                if p == seatWind:
                    # bugang, self
                    hand_list[t[3]] = 0
                    del hand_list[t[3]]
                    hand_list["shown"].append({t[3]: 4})
                    print("PASS")
                else:
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
                    # update tile_list
                    # check hu

                    hand_list_cp = copy.deepcopy(hand_list)
                    hand_list_cp[t[-1]] += 1
                    tile_list = rule.update_tile_info(tile_list, {t[-1]: 1})
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

                    (action, mid_tile, eject_tile, max_prob) = test_chi_peng(
                        hand_list_cp,
                        tile_list,
                        t[-1],
                        seatWind,
                        prevailing_wind,
                        test_chi,
                        play_efficiency_lb=1.5,
                        held_prob=defaultdict(default_zero),
                        discard_prob=defaultdict(default_zero),
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
