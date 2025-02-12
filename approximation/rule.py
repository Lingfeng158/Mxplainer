# 规则
import numpy as np
import os
# import MahjongGB
# from MahjongGB import MahjongFanCalculator
from collections import defaultdict


def default_zero():
    return 0


def one_type_tile_count(type, hand_list):
    """
    return how many pieces of the type
    """
    total = 0
    for rank in range(1, 10):
        key = type + str(rank)
        total += hand_list.get(key, 0)

    return total


def one_type_dist_to_straight(type, hand_list, tile_list):
    """
    return list of distance to straight
    某花色到顺子的所有距离
    2 to 8, being the middle tile
    """
    scratch_place = {}
    for rank in range(1, 10):
        key = type + str(rank)
        scratch_place[key] = {
            "hand": hand_list.get(key, 0),
            "remain": tile_list[key],
        }

    dist_to_straight = []
    avail_tiles_to_straight = []
    for rank in range(2, 9):
        midkey = type + str(rank)
        lowkey = type + str(rank - 1)
        highkey = type + str(rank + 1)

        mid_dist = (
            0
            if scratch_place[midkey]["hand"] > 0
            else 1
            if scratch_place[midkey]["remain"] > 0
            else 9
        )
        low_dist = (
            0
            if scratch_place[lowkey]["hand"] > 0
            else 1
            if scratch_place[lowkey]["remain"] > 0
            else 9
        )
        high_dist = (
            0
            if scratch_place[highkey]["hand"] > 0
            else 1
            if scratch_place[highkey]["remain"] > 0
            else 9
        )
        sum_dist = mid_dist + low_dist + high_dist
        dist_to_straight.append(sum_dist)
        if sum_dist == 0 or sum_dist > 3:
            avail_tiles_to_straight.append(0)
        else:
            temp_tile = 0
            if low_dist != 0:
                temp_tile += scratch_place[lowkey]["remain"]
            if mid_dist != 0:
                temp_tile += scratch_place[midkey]["remain"]
            if high_dist != 0:
                temp_tile += scratch_place[highkey]["remain"]
            avail_tiles_to_straight.append(temp_tile)

    # print("dis to straight list: ", dist_to_straight)
    # print("avail tiles to straight list: ", avail_tiles_to_straight)
    return (
        np.array(dist_to_straight),
        np.array(avail_tiles_to_straight),
    )


def dist_to_straight_single(type, rank, hand_list, tile_list):
    """
    return list of distance to straight
    某花色到顺子的所有距离
    2 to 8, being the middle tile
    """

    midkey = type + str(rank)
    lowkey = type + str(rank - 1)
    highkey = type + str(rank + 1)
    hand_list = defaultdict(int, hand_list)

    mid_dist = 0 if hand_list[midkey] > 0 else 1 if tile_list[midkey] > 0 else 9
    low_dist = 0 if hand_list[lowkey] > 0 else 1 if tile_list[lowkey] > 0 else 9
    high_dist = 0 if hand_list[highkey] > 0 else 1 if tile_list[highkey] > 0 else 9
    sum_dist = mid_dist + low_dist + high_dist
    dist_to_straight = sum_dist
    if sum_dist == 0 or sum_dist > 3:
        avail_tiles_to_straight = 0
    else:
        temp_tile = 0
        if low_dist != 0:
            temp_tile += tile_list[lowkey]
        if mid_dist != 0:
            temp_tile += tile_list[midkey]
        if high_dist != 0:
            temp_tile += tile_list[highkey]
        avail_tiles_to_straight = temp_tile
    # print("dis to straight list: ", dist_to_straight)
    # print("avail tiles to straight list: ", avail_tiles_to_straight)
    return (
        dist_to_straight,
        avail_tiles_to_straight,
    )


def one_type_dist_to_trio(type, hand_list, tile_list):
    """
    return list of distance to trio
    某花色到刻子的所有距离
    1 to 9
    """
    dist_to_trio = []
    avail_tiles_to_trio = []
    if type != "X":
        for rank in range(1, 10):
            key = type + str(rank)
            owned_count = hand_list.get(key, 0)
            dist = max(0, 3 - owned_count) if owned_count + tile_list[key] >= 3 else 9
            dist_to_trio.append(dist)
            avail_tiles_to_trio.append(tile_list[key])
    else:
        for exact_tile in ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]:
            owned_count = hand_list.get(exact_tile, 0)
            dist = (
                max(0, 3 - owned_count)
                if owned_count + tile_list[exact_tile] >= 3
                else 9
            )
            dist_to_trio.append(dist)
            avail_tiles_to_trio.append(tile_list[exact_tile])

    # print("dis to trio list: ", dist_to_trio)
    # print("avail tiles to trio list: ", avail_tiles_to_trio)
    return (
        np.array(dist_to_trio),
        np.array(avail_tiles_to_trio),
    )


def dist_to_trio_single(type, rank, hand_list, tile_list):
    """
    return list of distance to trio
    某花色到刻子的所有距离
    1 to 9
    """
    if type != "X":
        key = type + str(rank)
        owned_count = hand_list.get(key, 0)
        dist = max(0, 3 - owned_count) if owned_count + tile_list[key] >= 3 else 9
        dist_to_trio = dist

        if dist != 0:
            avail_tiles_to_trio = tile_list[key]
        else:
            avail_tiles_to_trio = 0
    else:
        exact_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
        exact_tile = exact_tile_list[rank - 1]
        owned_count = hand_list.get(exact_tile, 0)
        dist = (
            max(0, 3 - owned_count) if owned_count + tile_list[exact_tile] >= 3 else 9
        )
        dist_to_trio = dist

        if dist != 0:
            avail_tiles_to_trio = tile_list[exact_tile]

        else:
            avail_tiles_to_trio = 0

    # print("dis to trio list: ", dist_to_trio)
    # print("avail tiles to trio list: ", avail_tiles_to_trio)
    return (
        dist_to_trio,
        avail_tiles_to_trio,
    )


def one_type_dist_to_duo(type, hand_list, tile_list):
    """
    return list of distance to duo
    某花色到对子的所有距离
    type: BWTX, where X could be F or J
    """
    dist_to_duo = []
    avail_tiles_to_duo = []
    if type != "X":
        for rank in range(1, 10):
            key = type + str(rank)
            owned_count = hand_list.get(key, 0)
            dist = max(0, 2 - owned_count) if owned_count + tile_list[key] >= 2 else 9
            dist_to_duo.append(dist)
            avail_tiles_to_duo.append(tile_list[key])
    else:
        for exact_tile in ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]:
            owned_count = hand_list.get(exact_tile, 0)
            dist = (
                max(0, 2 - owned_count)
                if owned_count + tile_list[exact_tile] >= 2
                else 9
            )
            dist_to_duo.append(dist)
            avail_tiles_to_duo.append(tile_list[exact_tile])

    return (
        np.array(dist_to_duo),
        np.array(avail_tiles_to_duo),
    )


def dist_to_duo_single(type, rank, hand_list, tile_list):
    """
    return list of distance to duo
    某花色到对子的所有距离
    type: BWTX, where X could be F or J
    """
    if type != "X":
        key = type + str(rank)
        owned_count = hand_list.get(key, 0)
        dist = max(0, 2 - owned_count) if owned_count + tile_list[key] >= 2 else 9
        dist_to_duo = dist

        if dist != 0:
            avail_tiles_to_duo = tile_list[key]
        else:
            avail_tiles_to_duo = 0
    else:
        exact_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
        exact_tile = exact_tile_list[rank - 1]
        owned_count = hand_list.get(exact_tile, 0)
        dist = (
            max(0, 2 - owned_count) if owned_count + tile_list[exact_tile] >= 2 else 9
        )
        dist_to_duo = dist

        if dist != 0:
            avail_tiles_to_duo = tile_list[exact_tile]

        else:
            avail_tiles_to_duo = 0

    return (
        dist_to_duo,
        avail_tiles_to_duo,
    )


def one_type_min_dist_and_tile(tile_type, dist, avail_count, is_straight):
    """
    Helper function from tile_dist + tile_avail to min_tile_dist and the choice
    return min distance, and choice of tile for min dist
    从某花色手牌到某花色的(顺子或刻子/对子)最短距离
    """
    # 所有最短距离的index
    min_tile_dist_index = np.where(dist == dist.min())[0]
    # 所有最短距离所属的牌墙可用性
    avail_count = np.take(
        avail_count, min_tile_dist_index
    )  #  only tile availability of min distance, shape equal to min_tile_dist
    index_of_high_avail = np.argmax(avail_count)  # index of shortened list
    final_index = min_tile_dist_index[index_of_high_avail]
    if is_straight:
        return (
            dist[final_index],
            tile_type + str(final_index + 2),
            avail_count[index_of_high_avail],
        )
    else:
        if tile_type == "X":
            if final_index < 4:
                tile = "F" + str(final_index + 1)
            else:
                tile = "J" + str(final_index - 3)
        else:
            tile = tile_type + str(final_index + 1)
        return (
            dist[final_index],
            tile,
            avail_count[index_of_high_avail],
        )


def from_tile_selection_to_tile_composition(
    hand_list, tile_selection, is_straight, is_duo=False
):
    """
    从tile_choice到tile_composition的, 基于手牌的合理映射
    return: 手牌已有，手牌目标
    """
    tile_type = tile_selection[0]
    tile_rank = tile_selection[1]
    tile_compisition = {}
    tile_target = {}
    if is_straight:
        midkey = tile_type + str(int(tile_rank))
        lowkey = tile_type + str(int(tile_rank) - 1)
        highkey = tile_type + str(int(tile_rank) + 1)
        if hand_list.get(lowkey, 0) > 0:
            tile_compisition[lowkey] = 1
        if hand_list.get(midkey, 0) > 0:
            tile_compisition[midkey] = 1
        if hand_list.get(highkey, 0) > 0:
            tile_compisition[highkey] = 1
        tile_target[lowkey] = 1
        tile_target[midkey] = 1
        tile_target[highkey] = 1
        return tile_compisition, tile_target
    else:
        if is_duo:
            tile_ct = min(hand_list.get(tile_selection, 0), 2)
            if tile_ct != 0:
                tile_compisition[tile_selection] = tile_ct
            tile_target[tile_selection] = 2
            return tile_compisition, tile_target
        else:
            tile_ct = min(hand_list.get(tile_selection, 0), 3)
            if tile_ct != 0:
                tile_compisition[tile_selection] = tile_ct
            tile_target[tile_selection] = 3
            return tile_compisition, tile_target


def one_type_min_dist_to_any_triplet(
    tile_type, hand_list, tile_list, is_Feng_or_Jian=False
):
    """
    从某花色选择任意最短单3距离, 以及麻将选择
    """
    if is_Feng_or_Jian:
        (
            tile_dist_to_trio,
            tile_avail_to_trio,
        ) = one_type_dist_to_trio(tile_type, hand_list, tile_list)
        (trio_min_dist, trio_choice, trio_avail,) = one_type_min_dist_and_tile(
            tile_type,
            tile_dist_to_trio,
            tile_avail_to_trio,
            False,
        )
        return (
            trio_min_dist,
            from_tile_selection_to_tile_composition(hand_list, trio_choice, False),
            trio_avail,
        )
    else:
        (
            tile_dist_to_straight,
            tile_avail_to_straight,
        ) = one_type_dist_to_straight(tile_type, hand_list, tile_list)
        (
            straight_min_dist,
            straight_choice,
            straight_avail,
        ) = one_type_min_dist_and_tile(
            tile_type,
            tile_dist_to_straight,
            tile_avail_to_straight,
            True,
        )
        (
            tile_dist_to_trio,
            tile_avail_to_trio,
        ) = one_type_dist_to_trio(tile_type, hand_list, tile_list)
        (trio_min_dist, trio_choice, trio_avail,) = one_type_min_dist_and_tile(
            tile_type,
            tile_dist_to_trio,
            tile_avail_to_trio,
            False,
        )
        if trio_min_dist > straight_min_dist:
            return (
                straight_min_dist,
                from_tile_selection_to_tile_composition(
                    hand_list, straight_choice, True
                ),
                straight_avail,
            )
        else:
            return (
                trio_min_dist,
                from_tile_selection_to_tile_composition(hand_list, trio_choice, False),
                trio_avail,
            )


def one_type_min_dist_to_duo(tile_type, hand_list, tile_list):
    """
    从某花色选择任意最短对子的距离, 以及麻将选择
    """

    tile_dist, tile_avail = one_type_dist_to_duo(tile_type, hand_list, tile_list)
    # min_tile_dist_index = np.where(tile_dist == tile_dist.min())[0]
    # tile_avail = np.take(
    #     tile_avail, min_tile_dist_index
    # )  #  only tile availability of min distance, shape equal to min_tile_dist
    # index_of_high_avail = np.argmax(tile_avail)  # index of shortened list
    # final_index = min_tile_dist_index[index_of_high_avail]
    # return tile_dist[final_index], type + str(final_index + 1)
    duo_min_dist, duo_choice, duo_avail = one_type_min_dist_and_tile(
        tile_type, tile_dist, tile_avail, False
    )
    return (
        duo_min_dist,
        from_tile_selection_to_tile_composition(hand_list, duo_choice, False, True),
        duo_avail,
    )


def from_custom_to_canonical_encoding(custom_enc):
    """
    switch from custom encoding to PyMahJongGB encoding
    input: dict with count
    output: list
    """
    output_list = []
    for key in custom_enc:
        if key != "shown":
            inst = custom_enc[key]
            for _ in range(inst):
                output_list.append(key)
    return output_list


def from_canonical_to_custom_encoding(can_enc):
    """
    switch from PyMahJongGB encoding to custom encoding
    """
    output_dict = defaultdict(default_zero)
    for tile in can_enc:
        output_dict[tile] += 1
    return output_dict


def update_tile_info(tile_list, tile_appearance_list):
    """
    return new tile_list from given tile_list, and tile_appearance_list
    """
    tile_list_ret = tile_list.copy()
    pack = tile_appearance_list.get("shown", [])
    for entry in pack:
        for t in entry:
            if t != "AnGang":
                tile_list_ret[t] -= entry[t]
    for t in tile_appearance_list:
        if t != "shown":
            tile_list_ret[t] -= tile_appearance_list[t]
    return tile_list_ret


# def calc_fan_with_PyMahJongGB_sample(targ_enc, seat_wind=0, prevalentWind=1):
#     """
#     Sample test if proposed hand set satisfies eight-fan
#     """
#     fan_sum = 0
#     canonical_targ_list = []
#     for targ in targ_enc:
#         partial_targ_list = from_custom_to_canonical_encoding(targ)
#         canonical_targ_list += partial_targ_list
#     last_tile = canonical_targ_list[0]
#     hand_list = canonical_targ_list[1:]

#     ret = MahjongFanCalculator(
#         pack=(),
#         hand=tuple(hand_list),
#         winTile=last_tile,
#         flowerCount=0,
#         isSelfDrawn=False,
#         is4thTile=False,
#         isAboutKong=False,
#         isWallLast=False,
#         seatWind=seat_wind,
#         prevalentWind=prevalentWind,
#     )

#     return ret


# def calc_fan_with_PyMahJongGB(sel_enc, targ_enc, pack, seat_wind=0, prevalentWind=1):
#     """
#     Test if proposed hand set satisfies eight-fan
#     return: if_is_valid_composition, unrestricted_chi_peng (是否有吃，碰限制, 不计已有的手牌限制), unrestricted_he(是否有和牌限制),
#     valid_win_tile (如需要"边张", "嵌张", "单钓将"),  need_plus_one_dist(是否需要距离加一,  强行凑"边张", "嵌张", "单钓将"),
#     """
#     fan_sum = 0
#     canonical_targ_list = []
#     for targ in targ_enc:
#         partial_targ_list = from_custom_to_canonical_encoding(targ)
#         canonical_targ_list += partial_targ_list
#     canonical_hand_list = from_custom_to_canonical_encoding(sel_enc)
#     canonical_pack = from_custom_to_canonical_pack(pack)
#     last_tile_selection = []
#     min_fan = 1000
#     max_fan = 0
#     free_chi_peng = True
#     free_hu = True
#     for i in range(len(canonical_hand_list)):
#         last_tile = canonical_targ_list[i]
#         hand_list = canonical_targ_list[:i] + canonical_targ_list[i + 1 :]

#         ret = MahjongFanCalculator(
#             pack=(canonical_pack),
#             hand=tuple(hand_list),
#             winTile=last_tile,
#             flowerCount=0,
#             isSelfDrawn=False,
#             is4thTile=False,
#             isAboutKong=False,
#             isWallLast=False,
#             seatWind=seat_wind,
#             prevalentWind=prevalentWind,
#         )
#         fan_sum = 0
#         for fan in ret:
#             fan_sum += fan[0]
#         if fan_sum < min_fan:
#             min_fan = fan_sum
#         if fan_sum > max_fan:
#             max_fan = fan_sum
#         if fan_sum >= 8:
#             last_tile_selection.append(last_tile)

#         # rectify for chi/peng
#         rectified_fan = fan_sum
#         for fan in ret:
#             if fan[1] in ["不求人", "双暗刻", "三暗刻", "四暗刻", "双暗杠", "暗杠", "门前清"]:
#                 rectified_fan -= fan[0]
#         if rectified_fan < 8:
#             free_chi_peng = False

#         # rectify for hu
#         rectified_fan = fan_sum
#         for fan in ret:
#             if fan[1] in ["边张", "嵌张", "单钓将"]:
#                 rectified_fan -= fan[0]
#         if rectified_fan < 8:
#             free_hu = False

#     if min_fan >= 8 and free_chi_peng == True:
#         return (True, True, True, [], False)
#     if max_fan < 8:
#         return (False, False, False, [], False)
#     last_tile_selection = set(last_tile_selection)
#     canonical_hand_set = set(canonical_hand_list)
#     tile_diff = last_tile_selection - canonical_hand_set
#     if len(tile_diff) == 0:
#         # 全摸齐后，打掉手牌，凑"边张", "嵌张", "单钓将"
#         return (True, free_chi_peng, free_hu, last_tile_selection, True)
#     else:
#         # 有可用"边张", "嵌张", "单钓将"
#         return (True, free_chi_peng, free_hu, list(tile_diff), False)


# def calc_exact_fan_with_PyMahJongGB(
#     pack,
#     handWall,
#     win_tile,
#     is_last_tile,
#     is_self_drawn,
#     is_4th_tile,
#     is_kong_related,
#     seat_wind,
#     prevailingWind,
# ):
#     """
#     Adapter for fan calculator from PyMahJongGB
#     handWall + pack's typical count = 14
#     """
#     pack = from_custom_to_canonical_pack(pack)
#     handWall = from_custom_to_canonical_encoding(handWall)
#     handWall.remove(win_tile)
#     ret = MahjongFanCalculator(
#         pack=(pack),
#         hand=tuple(handWall),
#         winTile=win_tile,
#         flowerCount=0,
#         isSelfDrawn=is_self_drawn,
#         is4thTile=is_4th_tile,
#         isAboutKong=is_kong_related,
#         isWallLast=is_last_tile,
#         seatWind=seat_wind,
#         prevalentWind=prevailingWind,
#     )
#     fan_sum = 0
#     fan_list = []
#     for fan in ret:
#         fan_sum += fan[0]
#         fan_list.append(fan[1])
#     return fan_sum, fan_list


def check_hu(
    hand_wall,
    tile_wall,
    new_tile,
    seatwind,
    prevailing_wind,
    is_self_drawn,
    is_about_kong=False,
):
    """
    Check if hu using calc_exact_fan_with_PyMahJongGB
    """
    pack = hand_wall["shown"]

    # is 4th tile
    is_4th_tile = True
    if tile_wall[new_tile] > 0:
        is_4th_tile = False
    if hand_wall[new_tile] > 1:
        is_4th_tile = False

    # is last tile
    try:
        fan_sum, fan_list = calc_exact_fan_with_PyMahJongGB(
            pack,
            hand_wall,
            new_tile,
            False,
            is_self_drawn,
            is_4th_tile,
            is_about_kong,
            seatwind,
            prevailing_wind,
        )
        if fan_sum >= 8:
            return fan_sum
        else:
            return 0
    except Exception:
        return 0


# def calc_fan_with_PyMahJongGB_quick(
#     sel_enc, targ_enc, pack, seat_wind=0, prevalentWind=1, disable_juezhang=False
# ):
#     """
#     Test if proposed hand set satisfies eight-fan
#     only try from unfinished_set
#     return: if_is_valid_composition, unrestricted_chi_peng (是否有吃，碰限制, 不计已有的手牌限制), unrestricted_he(是否有和牌限制),
#     valid_win_tile (如需要"边张", "嵌张", "单钓将"), need_plus_one_dist(是否需要距离加一, 强行凑"边张", "嵌张", "单钓将"),
#     """
#     fan_sum = 0
#     # canonical_targ_list = []
#     # for targ in targ_enc:
#     #     partial_targ_list = from_custom_to_canonical_encoding(targ)
#     #     canonical_targ_list += partial_targ_list
#     canonical_complete_list, canonical_incomplete_list = seperate_incomplete_set(
#         sel_enc, targ_enc
#     )
#     canonical_pack = from_custom_to_canonical_pack(pack)
#     canonical_hand_list = from_custom_to_canonical_encoding(sel_enc)
#     last_tile_selection = []
#     min_fan = 1000
#     max_fan = 0
#     free_chi_peng = True
#     free_hu = True
#     if len(canonical_incomplete_list) == 0:
#         canonical_incomplete_list = canonical_complete_list
#         canonical_complete_list = []
#     for i in range(len(canonical_incomplete_list)):
#         last_tile = canonical_incomplete_list[i]
#         hand_list = (
#             canonical_incomplete_list[:i]
#             + canonical_incomplete_list[i + 1 :]
#             + canonical_complete_list
#         )

#         ret = MahjongFanCalculator(
#             pack=(canonical_pack),
#             hand=tuple(hand_list),
#             winTile=last_tile,
#             flowerCount=0,
#             isSelfDrawn=False,
#             is4thTile=False,
#             isAboutKong=False,
#             isWallLast=False,
#             seatWind=seat_wind,
#             prevalentWind=prevalentWind,
#         )
#         fan_sum = 0
#         for fan in ret:
#             if disable_juezhang and fan[1] == "和绝张":
#                 continue
#             fan_sum += fan[0]
#         if fan_sum < min_fan:
#             min_fan = fan_sum
#         if fan_sum > max_fan:
#             max_fan = fan_sum
#         if fan_sum >= 8:
#             last_tile_selection.append(last_tile)

#         # rectify for chi/peng
#         rectified_fan = fan_sum
#         for fan in ret:
#             if fan[1] in ["不求人", "双暗刻", "双暗杠", "暗杠", "门前清"]:
#                 rectified_fan -= fan[0]
#         if fan_sum >= 8 and rectified_fan < 8:
#             free_chi_peng = False

#         # rectify for hu
#         rectified_fan = fan_sum
#         for fan in ret:
#             if fan[1] in ["边张", "嵌张", "单钓将"]:
#                 rectified_fan -= fan[0]
#         if rectified_fan < 8:
#             free_hu = False
#     if min_fan >= 8 and free_chi_peng == True:
#         return (True, True, True, [])
#     if max_fan < 8:
#         return (False, False, False, [])

#     return (True, free_chi_peng, free_hu, list(set(last_tile_selection)))


def seperate_incomplete_set(sel_enc, targ_enc):
    """
    seperate incomplete set from complete ones (duo, trio, straight)
    return complete set, incomplete set
    """
    canonical_hand_list = from_custom_to_canonical_encoding(sel_enc)
    canonical_incomplete_list = []
    canonical_complete_list = []
    # canonical_sample_complete_set = []
    for targ in targ_enc:
        set_complete = True
        # is_trio = False
        partial_targ_list = from_custom_to_canonical_encoding(targ)
        for t in partial_targ_list:
            # if targ[t] == 3:
            #     is_trio == True
            if t not in canonical_hand_list:
                set_complete = False
            if t in canonical_hand_list:
                canonical_hand_list.remove(t)
        if set_complete == True:
            canonical_complete_list += partial_targ_list
        else:
            canonical_incomplete_list += partial_targ_list
        # if set_complete == True and is_trio == True:
        #     canonical_sample_complete_set = partial_targ_list

    return (
        # canonical_sample_complete_set,
        canonical_complete_list,
        canonical_incomplete_list,
    )


# handle pack 附录


def from_canonical_to_custom_pack(canonical_pack):
    """
    from PyMahJongGB's canonical representation to custom representation
    return [{},{}], list of dicts
    """
    ret = []
    for entry in canonical_pack:
        if entry[0] == "PENG":
            ret.append({entry[1]: 3})
        elif entry[0] == "GANG":
            ret.append({entry[1]: 4})
        elif entry[0] == "CHI":
            tile_type = entry[1][0]
            tile_rank = int(entry[1][1])
            ret.append(
                {
                    tile_type + str(tile_rank - 1): 1,
                    entry[1]: 1,
                    tile_type + str(tile_rank + 1): 1,
                }
            )
        else:
            raise TypeError("ERROR in canonical pack")
    return ret


def from_custom_to_canonical_pack(custom_pack):
    """
    from custom pack representation to PyMahJongGB's canonical representation
    "offer" defaults to 1, which is left hand side player as place holder
    return ((packType, tileCode, offer), ...)
    """
    ret = []
    for entry in custom_pack:
        if len(entry) > 2:
            for key in entry:
                # find midkey
                tile_type = key[0]
                tile_rank = int(key[1])
                # only true for midkey
                if (
                    entry.get(tile_type + str(tile_rank + 1), 0) == 1
                    and entry.get(tile_type + str(tile_rank - 1), 0) == 1
                ):
                    ret.append(("CHI", key, 1))
        else:
            for key in entry:
                if key == "AnGang":
                    continue
                if entry[key] == 3:
                    ret.append(("PENG", key, 1))
                elif entry[key] == 4:
                    if "AnGang" in entry and entry["AnGang"] == True:
                        ret.append(("GANG", key, 0))
                    else:
                        ret.append(("GANG", key, 1))
    return tuple(ret)


def hash_custom_tiles(custom_tile_dict):
    """
    hash custom encoded tile list into string, encoding alg: 34 digits of tiles, each digit represent tile count, order: BWTFJ
    """
    result_list = [0] * 34
    dict_order = [
        *("B%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i + 1) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ]
    for key in custom_tile_dict:
        if key != "shown":
            dict_id = dict_order.index(key)
            val = custom_tile_dict[key]
            result_list[dict_id] += val
    return "".join(str(e) for e in result_list)


def restore_hashed_tiles(tile_hash):
    """
    restore custom encoding from hash, order: BWTFJ
    """
    dict_order = [
        *("B%d" % (i + 1) for i in range(9)),
        *("W%d" % (i + 1) for i in range(9)),
        *("T%d" % (i + 1) for i in range(9)),
        *("F%d" % ((i + 1) % 4 + 1) for i in range(4)),
        *("J%d" % ((i + 1) % 3 + 1) for i in range(3)),
    ]
    result_dict = defaultdict(default_zero)
    for i in range(34):
        if int(tile_hash[i]) != 0:
            result_dict[dict_order[i]] = int(tile_hash[i])
    return result_dict


def hash_seperated_custom_tile(custom_tile_dict_in_list):
    """
    same function as "hash_custom_tile" function, but dicts are in list, i.e. [{...},{...}, ...]
    flatten dicts and then call "hash_custom_tile"
    """
    # print(custom_tile_dict_in_list)
    # print(custom_tile_dict_in_list)
    flatten_dict = defaultdict(default_zero)
    isHepta = True
    for tile_dict in custom_tile_dict_in_list:
        for k in tile_dict:
            if k != "AnGang" and tile_dict[k] % 2 == 1:
                isHepta = False

    for tile_dict in custom_tile_dict_in_list:
        for k in tile_dict:
            # print(tile_dict)
            # print("key", k)
            # print(tile_dict[k])
            if k != "AnGang":
                flatten_dict[k] += (
                    tile_dict[k] if isHepta else min(tile_dict[k], 3)
                )  # disregard Gang
    return hash_custom_tiles(flatten_dict)


# 上听数，由PyMahJongGB包装
def PyMajJongGB_shanten(hand_list, pack_list):
    pack = from_custom_to_canonical_pack(pack_list)
    hand = tuple(from_custom_to_canonical_encoding(hand_list))
    shanten_count = MahjongGB.MahjongShanten(pack, hand)
    if shanten_count == 0:
        return True
    return False
