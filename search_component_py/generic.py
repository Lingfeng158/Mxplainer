import sys
import os

sys.path.append("/data")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import numpy as np
import src_py.rule
import src_py.special
import copy
from collections import defaultdict


def one_type_all_combination(
    tile_type,
    hand_list,
    tile_list,
    triplet_count,
    duo_count,
    max_dist=7,
    total_iteration=0,
    passed_on_dist=0,
):
    """
    生成不同组合的, 最低距离的DP表
    tile_type: BWTX, where X represents F and J
    hand_list: 手牌
    tile_list: 记录每张牌的可用性
    triple_count: 需要多少个顺子、刻子
    duo_count: 需要多少个对子
    max_dist: 由greedy search产生的prior knowledge, 或者从上级继承, 用于prunning
    passed_on_dist: 由上级继承, 默认0, 用于prunning
    return: min_dist, select_tiles, tile_availability
    """
    # print(tile_list)
    # pre filter
    total_req_tiles = triplet_count * 3 + duo_count * 2 - total_iteration * 3
    total_hold_tiles = src_py.rule.one_type_tile_count(tile_type, hand_list)
    # print(total_hold_tiles, total_req_tiles)
    if total_req_tiles - total_hold_tiles > max_dist:

        return [], 0, {}, 0, [], []

    min_dist = max_dist
    selected_tiles = {}
    target_tiles = []
    tiles_availability = 0
    tile_list_cp = tile_list.copy()
    ret_list = []
    for temp_rank in range(1, 10 + 7):
        hand_list_cp = hand_list.copy()
        rank = temp_rank
        if rank < 10:
            if tile_type != "X":
                if total_iteration < triplet_count:
                    # trio
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.one_type_dist_to_trio(
                        tile_type, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), False
                    )
                else:
                    # duo
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.one_type_dist_to_duo(
                        tile_type, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), False, True
                    )
                temp_dist = temp_dist[rank - 1]
                temp_avail = temp_avail[rank - 1]
            else:
                continue
        else:
            if tile_type != "X":
                rank -= 8
                if total_iteration < triplet_count:
                    # straight
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.one_type_dist_to_straight(
                        tile_type, hand_list, tile_list_cp
                    )
                else:
                    # duo, does not exist, pass on
                    continue
                (
                    temp_tile_compo,
                    temp_tile_targ,
                ) = src_py.rule.from_tile_selection_to_tile_composition(
                    hand_list_cp, tile_type + str(rank), True
                )
                temp_dist = temp_dist[rank - 2]
                temp_avail = temp_avail[rank - 2]
            else:
                # tile type == X
                special_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
                rank -= 10
                if total_iteration < triplet_count:
                    # trio
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.one_type_dist_to_trio(
                        tile_type, hand_list_cp, tile_list_cp
                    )

                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank], False
                    )
                else:
                    # duo
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.one_type_dist_to_duo(
                        tile_type, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank], False, True
                    )
                temp_dist = temp_dist[rank]
                temp_avail = temp_avail[rank]

        # 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
        if temp_dist + passed_on_dist > max_dist:
            continue

        # 减枝未发生，更新手牌信息
        for each_tile in temp_tile_compo:
            hand_list_cp[each_tile] -= temp_tile_compo[each_tile]

        # 更新可用牌墙信息
        tile_list_cp_update = src_py.rule.update_tile_info(tile_list_cp, temp_tile_targ)

        if total_iteration + 1 != triplet_count + duo_count:
            (
                _,
                ret_dist,
                ret_tiles,
                ret_avail,
                ret_targ,
            ) = one_type_all_combination(
                tile_type,
                hand_list_cp,
                tile_list_cp_update,
                triplet_count,
                duo_count,
                max_dist,
                total_iteration + 1,
                temp_dist,
            )
        else:
            # terminal situation
            ret_dist, ret_tiles, ret_avail, ret_targ = (
                0,
                {},
                0,
                [],
            )

        calculated_dist = temp_dist + ret_dist

        if calculated_dist > max_dist:
            continue
        else:
            # 整合 selected tiles
            for t in temp_tile_compo:
                if t not in ret_tiles:
                    ret_tiles[t] = temp_tile_compo[t]
                else:
                    ret_tiles[t] += temp_tile_compo[t]

            # ret_targ_list = []
            # for t in temp_tile_targ:
            #     if t not in ret_targ:
            #         ret_targ[t] = temp_tile_targ[t]
            #     else:
            #         ret_targ[t] += temp_tile_targ[t]
            ret_targ.append(temp_tile_targ)
            # if tile_type == "T":
            #     print(
            #         triplet_count,
            #         duo_count,
            #         total_iteration,
            #         ret_targ,
            #     )
            # 整合 tiles_avail
            ret_avail += temp_avail

            # 所有符合要求的排列组合
            # 对要求进行检查
            # print(calculated_dist, ret_tiles, ret_avail, ret_avail_detail, ret_targ)
            if len(ret_targ) == triplet_count + duo_count:
                ret_list.append([calculated_dist, ret_tiles, ret_avail, ret_targ])

        if (
            calculated_dist < min_dist
            or calculated_dist == min_dist
            and tiles_availability < ret_avail
        ):
            min_dist = calculated_dist
            selected_tiles = ret_tiles
            tiles_availability = ret_avail
            target_tiles = ret_targ
    return (
        ret_list,
        min_dist,
        selected_tiles,
        tiles_availability,
        target_tiles,
    )


def one_type_all_combination_dp(
    tile_type,
    hand_list,
    tile_list,
    triplet_count,
    duo_count,
    lookup_table=None,
    max_dist=6,
):
    """
    生成不同组合的, 最低距离的DP表
    tile_type: BWTX, where X represents F and J
    hand_list: 手牌
    tile_list: 记录每张牌的可用性
    triple_count: 需要多少个顺子、刻子
    duo_count: 需要多少个对子
    lookup_table: 由上级调用此函数产生的lookup_table, 包含（min_dist, select_tiles, tile_availability, target_tiles, hand_list_mod, tile_list_mod）
    max_dist: 由greedy search产生的prior knowledge, 或者从上级继承, 用于prunning
    return: lookup_table
    """
    prune_count = 12
    hash_key_list = []
    if triplet_count + duo_count == 0:
        return None
    if lookup_table == None and triplet_count + duo_count != 1:
        print(
            "ERROR, CALL THIS FUNC FROM TRIPLET_COUNT = 1 && DUO_COUNT = 0 OR TRIPLET_COUNT = 0 && DUO_COUNT = 1"
        )
    if triplet_count == 1 and duo_count == 0:
        # create lookup_table
        if lookup_table == None:
            lookup_table = {tile_type + "T1D0": []}
        else:
            lookup_table[tile_type + "T1D0"] = []
        lookup_page = []
        # Use passed in hand_list and tile_list
        for temp_rank in range(1, 10 + 7):
            hand_list_cp = copy.deepcopy(hand_list)
            tile_list_cp = tile_list.copy()
            rank = temp_rank
            if rank < 10:
                # not F or J
                if tile_type != "X":
                    # trios
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_trio_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), False
                    )
                else:
                    # does not work for F or J
                    continue
            else:
                if tile_type != "X":
                    rank -= 8
                    # straight
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_straight_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), True
                    )
                else:
                    # tile type == X
                    special_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
                    rank -= 9
                    # trio
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_trio_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )

                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank - 1], False
                    )
            passed_on_dist = 0
            # 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
            if temp_dist + passed_on_dist > max_dist:
                continue

            # 减枝未发生，更新手牌信息
            for each_tile in temp_tile_compo:
                hand_list_cp[each_tile] -= temp_tile_compo[each_tile]

            # 计算需更新牌墙信息
            missing_tile = {}
            for each_tile in temp_tile_targ:
                missing_tile[each_tile] = temp_tile_targ[
                    each_tile
                ] - temp_tile_compo.get(each_tile, 0)

            # 更新可用牌墙信息
            tile_list_cp_update = src_py.rule.update_tile_info(
                tile_list_cp, missing_tile
            )
            lookup_page.append(
                [
                    temp_dist,
                    temp_tile_compo,
                    temp_avail,
                    [temp_tile_targ],
                    hand_list_cp,
                    tile_list_cp_update,
                ]
            )
        # prunning: remove high dist entries if there are more than enough entries
        if len(lookup_page) <= prune_count:
            lookup_table[tile_type + "T1D0"] = lookup_page
            return lookup_table
        dist_dict = defaultdict(int)
        for ent in lookup_page:
            dist_dict[ent[0]] += 1
        total_entry_count = 0
        cut_dist = 9
        for i in range(max_dist):
            total_entry_count += dist_dict[i]
            if total_entry_count >= prune_count:
                cut_dist = i
                break
        for ent in lookup_page:
            if ent[0] <= cut_dist:
                lookup_table[tile_type + "T1D0"].append(ent)
        return lookup_table
    elif triplet_count == 0 and duo_count == 1:
        # create lookup_table
        if lookup_table == None:
            lookup_table = {tile_type + "T0D1": []}
        else:
            lookup_table[tile_type + "T0D1"] = []
        lookup_page = []
        # Use passed in hand_list and tile_list
        for temp_rank in range(1, 10):
            rank = temp_rank
            hand_list_cp = copy.deepcopy(hand_list)
            tile_list_cp = tile_list.copy()
            if tile_type != "X":
                # duo
                (
                    temp_dist,
                    temp_avail,
                ) = src_py.rule.dist_to_duo_single(
                    tile_type, rank, hand_list_cp, tile_list_cp
                )
                (
                    temp_tile_compo,
                    temp_tile_targ,
                ) = src_py.rule.from_tile_selection_to_tile_composition(
                    hand_list_cp, tile_type + str(rank), False, True
                )
            else:
                special_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
                if rank < 8:
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_duo_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank - 1], False, True
                    )
                else:
                    # does not do anything
                    continue

            passed_on_dist = 0
            # 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
            if temp_dist + passed_on_dist > max_dist:
                continue

            # 减枝未发生，更新手牌信息
            for each_tile in temp_tile_compo:
                hand_list_cp[each_tile] -= temp_tile_compo[each_tile]

            # 计算需更新牌墙信息
            missing_tile = {}
            for each_tile in temp_tile_targ:
                missing_tile[each_tile] = temp_tile_targ[
                    each_tile
                ] - temp_tile_compo.get(each_tile, 0)

            # 更新可用牌墙信息
            tile_list_cp_update = src_py.rule.update_tile_info(
                tile_list_cp, missing_tile
            )
            lookup_page.append(
                [
                    temp_dist,
                    temp_tile_compo,
                    temp_avail,
                    [temp_tile_targ],
                    hand_list_cp,
                    tile_list_cp_update,
                ]
            )
        # prunning: remove high dist entries if there are more than enough entries
        if len(lookup_page) <= prune_count:
            lookup_table[tile_type + "T0D1"] = lookup_page
            return lookup_table
        dist_dict = defaultdict(int)
        for ent in lookup_page:
            dist_dict[ent[0]] += 1
        total_entry_count = 0
        cut_dist = 9
        for i in range(max_dist):
            total_entry_count += dist_dict[i]
            if total_entry_count >= prune_count:
                cut_dist = i
                break
        for ent in lookup_page:
            if ent[0] <= cut_dist:
                lookup_table[tile_type + "T0D1"].append(ent)
        return lookup_table
    else:

        # use lookup_table and search triplets
        # load lookup_table list
        key = tile_type + "T{}D{}".format(triplet_count - 1, duo_count)
        lookup_list = lookup_table[key]
        # create new entry in lookup_table
        lookup_table[tile_type + "T{}D{}".format(triplet_count, duo_count)] = []
        lookup_page = []

        for lookup_entry in lookup_list:
            passed_on_dist = lookup_entry[0]
            passed_on_compo = lookup_entry[1]
            passed_on_avail = lookup_entry[2]
            passed_on_targ = lookup_entry[3]
            hand_l = lookup_entry[4]
            tile_l = lookup_entry[5]
            # # early stop
            total_req_tiles = 3
            total_hold_tiles = src_py.rule.one_type_tile_count(tile_type, hand_l)
            # print(total_hold_tiles, total_req_tiles)
            if total_req_tiles - total_hold_tiles + passed_on_dist > max_dist:
                continue

            for temp_rank in range(1, 10 + 7):
                hand_list_cp = copy.deepcopy(hand_l)
                tile_list_cp = tile_l.copy()
                rank = temp_rank
                if rank < 10:
                    # not F or J
                    if tile_type != "X":
                        # trios
                        (
                            temp_dist,
                            temp_avail,
                        ) = src_py.rule.dist_to_trio_single(
                            tile_type, rank, hand_list_cp, tile_list_cp
                        )
                        (
                            temp_tile_compo,
                            temp_tile_targ,
                        ) = src_py.rule.from_tile_selection_to_tile_composition(
                            hand_list_cp, tile_type + str(rank), False
                        )
                    else:
                        # does not work for F or J
                        continue
                else:
                    if tile_type != "X":
                        rank -= 8
                        # straight
                        (
                            temp_dist,
                            temp_avail,
                        ) = src_py.rule.dist_to_straight_single(
                            tile_type, rank, hand_list_cp, tile_list_cp
                        )
                        (
                            temp_tile_compo,
                            temp_tile_targ,
                        ) = src_py.rule.from_tile_selection_to_tile_composition(
                            hand_list_cp, tile_type + str(rank), True
                        )
                    else:
                        # tile type == X
                        special_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
                        rank -= 9
                        # trio
                        (
                            temp_dist,
                            temp_avail,
                        ) = src_py.rule.dist_to_trio_single(
                            tile_type, rank, hand_list_cp, tile_list_cp
                        )

                        (
                            temp_tile_compo,
                            temp_tile_targ,
                        ) = src_py.rule.from_tile_selection_to_tile_composition(
                            hand_list_cp, special_tile_list[rank - 1], False
                        )
                # 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
                if temp_dist + passed_on_dist > max_dist:
                    continue

                # 减枝未发生，更新手牌信息
                for each_tile in temp_tile_compo:
                    hand_list_cp[each_tile] -= temp_tile_compo[each_tile]

                ret_tiles = passed_on_compo.copy()
                for t in temp_tile_compo:
                    if t not in ret_tiles:
                        ret_tiles[t] = temp_tile_compo[t]
                    else:
                        ret_tiles[t] += temp_tile_compo[t]

                ret_targ = copy.deepcopy(passed_on_targ)
                ret_targ.append(temp_tile_targ)

                # hash target_tiles to avoid duplications
                h = src_py.rule.hash_seperated_custom_tile(ret_targ)
                if h not in hash_key_list:
                    hash_key_list.append(h)
                else:
                    continue

                # 计算需更新牌墙信息
                missing_tile = {}
                for each_tile in temp_tile_targ:
                    missing_tile[each_tile] = temp_tile_targ[
                        each_tile
                    ] - temp_tile_compo.get(each_tile, 0)

                # 更新可用牌墙信息
                tile_list_cp_update = src_py.rule.update_tile_info(
                    tile_list_cp, missing_tile
                )
                lookup_page.append(
                    [
                        temp_dist + passed_on_dist,
                        ret_tiles,
                        temp_avail + passed_on_avail,
                        ret_targ,
                        hand_list_cp,
                        tile_list_cp_update,
                    ]
                )
        # prunning: remove high dist entries if there are more than enough entries
        if len(lookup_page) <= prune_count:
            lookup_table[tile_type + "T{}D{}".format(triplet_count, duo_count)] = (
                lookup_page
            )
            return lookup_table
        dist_dict = defaultdict(int)
        for ent in lookup_page:
            dist_dict[ent[0]] += 1
        total_entry_count = 0
        cut_dist = 9
        for i in range(max_dist):
            total_entry_count += dist_dict[i]
            if total_entry_count >= prune_count:
                cut_dist = i
                break
        for ent in lookup_page:
            if ent[0] <= cut_dist:
                lookup_table[
                    tile_type + "T{}D{}".format(triplet_count, duo_count)
                ].append(ent)
        return lookup_table


def one_type_all_combination_alt(
    tile_type,
    hand_list,
    tile_list,
    triplet_count,
    duo_count,
    max_dist=7,
    total_iteration=0,
    passed_on_dist=0,
):
    """
    生成不同组合的, 最低距离的DP表
    tile_type: BWTX, where X represents F and J
    hand_list: 手牌
    tile_list: 记录每张牌的可用性
    triple_count: 需要多少个顺子、刻子
    duo_count: 需要多少个对子
    max_dist: 由greedy search产生的prior knowledge, 或者从上级继承, 用于prunning
    passed_on_dist: 由上级继承, 默认0, 用于prunning
    return: min_dist, select_tiles, tile_availability
    """
    # print(tile_list)
    # pre filter
    total_req_tiles = triplet_count * 3 + duo_count * 2 - total_iteration * 3
    total_hold_tiles = src_py.rule.one_type_tile_count(tile_type, hand_list)
    # print(total_hold_tiles, total_req_tiles)
    if total_req_tiles - total_hold_tiles > max_dist:

        return [], 0, {}, 0, []

    min_dist = max_dist
    selected_tiles = {}
    target_tiles = []
    tiles_availability = 0
    tile_list_cp = tile_list.copy()
    ret_list = []
    for temp_rank in range(1, 10 + 7):
        hand_list_cp = hand_list.copy()
        rank = temp_rank
        if rank < 10:
            if tile_type != "X":
                if total_iteration < triplet_count:
                    # trio
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_trio_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), False
                    )
                else:
                    # duo
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_duo_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, tile_type + str(rank), False, True
                    )
            else:
                continue
        else:
            if tile_type != "X":
                rank -= 8
                if total_iteration < triplet_count:
                    # straight
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_straight_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                else:
                    # duo, does not exist, pass on
                    continue
                (
                    temp_tile_compo,
                    temp_tile_targ,
                ) = src_py.rule.from_tile_selection_to_tile_composition(
                    hand_list_cp, tile_type + str(rank), True
                )
            else:
                # tile type == X
                special_tile_list = ["F1", "F2", "F3", "F4", "J1", "J2", "J3"]
                rank -= 9
                if total_iteration < triplet_count:
                    # trio
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_trio_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )

                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank - 1], False
                    )
                else:
                    # duo
                    (
                        temp_dist,
                        temp_avail,
                    ) = src_py.rule.dist_to_duo_single(
                        tile_type, rank, hand_list_cp, tile_list_cp
                    )
                    (
                        temp_tile_compo,
                        temp_tile_targ,
                    ) = src_py.rule.from_tile_selection_to_tile_composition(
                        hand_list_cp, special_tile_list[rank - 1], False, True
                    )

        # 减枝, 如果未循环到底部，但距离已经大于已知的min_dist
        if temp_dist + passed_on_dist > max_dist:
            continue

        # 减枝未发生，更新手牌信息
        for each_tile in temp_tile_compo:
            hand_list_cp[each_tile] -= temp_tile_compo[each_tile]

        # 计算需更新牌墙信息
        missing_tile = {}
        for each_tile in temp_tile_targ:
            missing_tile[each_tile] = temp_tile_targ[each_tile] - temp_tile_compo.get(
                each_tile, 0
            )

        # 更新可用牌墙信息
        tile_list_cp_update = src_py.rule.update_tile_info(tile_list_cp, missing_tile)

        if total_iteration + 1 != triplet_count + duo_count:
            (
                _,
                ret_dist,
                ret_tiles,
                ret_avail,
                ret_targ,
            ) = one_type_all_combination_alt(
                tile_type,
                hand_list_cp,
                tile_list_cp_update,
                triplet_count,
                duo_count,
                max_dist,
                total_iteration + 1,
                temp_dist,
            )
        else:
            # terminal situation
            ret_dist, ret_tiles, ret_avail, ret_targ = (
                0,
                {},
                0,
                [],
            )
        calculated_dist = temp_dist + ret_dist

        if calculated_dist > max_dist:
            continue
        if calculated_dist <= max_dist:
            # 整合 selected tiles
            for t in temp_tile_compo:
                if t not in ret_tiles:
                    ret_tiles[t] = temp_tile_compo[t]
                else:
                    ret_tiles[t] += temp_tile_compo[t]

            # ret_targ_list = []
            # for t in temp_tile_targ:
            #     if t not in ret_targ:
            #         ret_targ[t] = temp_tile_targ[t]
            #     else:
            #         ret_targ[t] += temp_tile_targ[t]
            ret_targ.append(temp_tile_targ)
            # if tile_type == "T":
            #     print(
            #         triplet_count,
            #         duo_count,
            #         total_iteration,
            #         ret_targ,
            #     )
            # 整合 tiles_avail
            ret_avail += temp_avail

            # 所有符合要求的排列组合
            # 对要求进行检查
            # print(calculated_dist, ret_tiles, ret_avail, ret_avail_detail, ret_targ)
            if len(ret_targ) == triplet_count + duo_count:
                ret_list.append([calculated_dist, ret_tiles, ret_avail, ret_targ])

        if (
            calculated_dist < min_dist
            or calculated_dist == min_dist
            and tiles_availability < ret_avail
        ):
            min_dist = calculated_dist
            selected_tiles = ret_tiles
            tiles_availability = ret_avail
            target_tiles = ret_targ
    return (
        ret_list,
        min_dist,
        selected_tiles,
        tiles_availability,
        target_tiles,
    )


def one_type_lookup_dp(tile_type, hand_list, tile_list):
    """
    Create a lookup chart for one type
    pos 0:min_dist
    pos 1:selected_tiles
    pos 2:tiles_availability
    pos 3:target_tiles
    """
    entry_in_pack = len(hand_list["shown"])
    max_allowed_dist_dict = {2: 1, 3: 2, 5: 3, 6: 4, 8: 5, 9: 5, 11: 5, 12: 6, 14: 6}
    lookup_table = {}
    lookup_table[tile_type + "T0D0"] = [(0, {}, 0, [], {}, {})]
    fill_default_0 = False
    fill_default_1 = False
    for triple_count in range(0, 5 - entry_in_pack):
        if triple_count != 0:
            if fill_default_0 == False:
                lookup_table = one_type_all_combination_dp(
                    tile_type,
                    hand_list,
                    tile_list,
                    triple_count,
                    0,
                    lookup_table,
                    max_dist=max_allowed_dist_dict[triple_count * 3],
                )
                if len(lookup_table[tile_type + "T" + str(triple_count) + "D0"]) == 0:
                    fill_default_0 = True
            else:
                lookup_table[tile_type + "T" + str(triple_count) + "D0"] = []
        if fill_default_1 == False:
            lookup_table = one_type_all_combination_dp(
                tile_type,
                hand_list,
                tile_list,
                triple_count,
                1,
                lookup_table,
                max_dist=max_allowed_dist_dict[triple_count * 3 + 2],
            )
            if len(lookup_table[tile_type + "T" + str(triple_count) + "D1"]) == 0:
                fill_default_1 = True
        else:
            lookup_table[tile_type + "T" + str(triple_count) + "D1"] = []
    return lookup_table


def one_type_lookup(tile_type, hand_list, tile_list):
    """
    Create a lookup chart for one type
    pos 0:min_dist
    pos 1:selected_tiles
    pos 2:tiles_availability
    pos 3:target_tiles
    """
    lookup = {}
    maximum_allowed_dist = 4
    # represent how many formed triplets
    entry_in_pack = len(hand_list["shown"])
    ret, _, _, _, _ = one_type_all_combination_alt(
        tile_type,
        hand_list,
        tile_list,
        0,
        1,
        max_dist=maximum_allowed_dist,
        total_iteration=0,
        passed_on_dist=0,
    )
    lookup[tile_type + "T0D0"] = [(0, {}, 0, [], {})]
    lookup[tile_type + "T0D1"] = ret
    fill_default_0 = False
    fill_default_1 = False
    for triplet_ct in range(1, 5 - entry_in_pack):
        if triplet_ct != 4 - entry_in_pack:
            max_dist = maximum_allowed_dist
        else:
            max_dist = maximum_allowed_dist + 1
        if fill_default_0 == False:
            ret0, _, _, _, _ = one_type_all_combination_alt(
                tile_type,
                hand_list,
                tile_list,
                triplet_ct,
                0,
                max_dist=max_dist,
                total_iteration=0,
                passed_on_dist=0,
            )
            lookup[tile_type + "T" + str(triplet_ct) + "D0"] = ret0
            if len(ret0) == 0:
                fill_default_0 = True
        else:
            lookup[tile_type + "T" + str(triplet_ct) + "D0"] = []
        if fill_default_1 == False:
            (
                ret1,
                _,
                _,
                _,
                _,
            ) = one_type_all_combination_alt(
                tile_type,
                hand_list,
                tile_list,
                triplet_ct,
                1,
                max_dist=max_dist,
                total_iteration=0,
                passed_on_dist=0,
            )
            lookup[tile_type + "T" + str(triplet_ct) + "D1"] = ret1
            if len(ret1) == 0:
                fill_default_1 = True
        else:
            lookup[tile_type + "T" + str(triplet_ct) + "D1"] = []

    return lookup


def form_min_combination(
    hand_list,
    tile_list,
    seat_wind=0,
    prevalentWind=1,
    result_threshold=15,
    max_dist=6,
    target_fan_val=8,
    disable_juezhang=False,
):
    """
    return 4 levels of list of possible hu combinations
    unrestricted: no limitation on chi/peng and hu win-tile
    restricted lv1: limitation on hu win-tile
    restricted lv2; limitation on chi/peng
    restricted lv3: limitation on both chi/peng and hu win-tile
    for each list: last element is the index in the list that points to shortest path to hu
    """
    count_available = 0
    look_up_list = []
    # represent how many formed triplets
    if isinstance(hand_list["shown"], list):
        entry_in_pack = len(hand_list["shown"])
    else:
        entry_in_pack = 0

    unrestricted_list = []  # no restriction on chi/peng, he
    unrestricted_min_dist = 10
    unrestricted_max_avail = 0
    unrestricted_id = 0
    restricted_lv1_list = []  # restriction on he
    restricted_lv1_min_dist = 10
    restricted_lv1_max_avail = 0
    restricted_lv1_id = 0
    restricted_lv2_list = []  # restriction on chi/peng
    restricted_lv2_min_dist = 10
    restricted_lv2_max_avail = 0
    restricted_lv2_id = 0
    restricted_lv3_list = []  # restriction on chi/peng and he
    restricted_lv3_min_dist = 10
    restricted_lv3_max_avail = 0
    restricted_lv3_id = 0
    for t in ["B", "W", "T", "X"]:
        lookup = one_type_lookup_dp(t, hand_list, tile_list)
        look_up_list.append(lookup)
    for try_dist in range(0, max_dist + 1):
        for b_key in look_up_list[0]:
            for w_key in look_up_list[1]:
                for t_key in look_up_list[2]:
                    for x_key in look_up_list[3]:
                        if (
                            int(b_key[2])
                            + int(w_key[2])
                            + int(t_key[2])
                            + int(x_key[2])
                            == 4 - entry_in_pack
                            and int(b_key[4])
                            + int(w_key[4])
                            + int(t_key[4])
                            + int(x_key[4])
                            == 1
                        ):
                            # print(entry_in_pack)
                            # print(b_key, w_key, t_key, x_key)
                            b = look_up_list[0][b_key]
                            w = look_up_list[1][w_key]
                            t = look_up_list[2][t_key]
                            x = look_up_list[3][x_key]
                            # print(t)

                            if len(b) == 0 or len(w) == 0 or len(t) == 0 or len(x) == 0:
                                continue

                            if count_available >= result_threshold:
                                break

                            # form union
                            for bb in b:
                                for ww in w:
                                    for tt in t:
                                        for xx in x:
                                            dist = bb[0] + ww[0] + tt[0] + xx[0]
                                            if dist == try_dist:
                                                # print(b[0], w[0], t[0])
                                                # return
                                                availability = (
                                                    bb[2] + ww[2] + tt[2] + xx[2]
                                                )

                                                # availability_detail = (
                                                #     bb[3] + ww[3] + tt[3] + xx[3]
                                                # )
                                                tiles = {}
                                                tiles.update(bb[1])
                                                tiles.update(ww[1])
                                                tiles.update(tt[1])
                                                tiles.update(xx[1])

                                                # sum planned tiles
                                                planned_tiles = []
                                                planned_tiles += bb[3]
                                                planned_tiles += ww[3]
                                                planned_tiles += tt[3]
                                                planned_tiles += xx[3]
                                                # print("PLANNED")
                                                # print(planned_tiles)

                                                # ===========HERE c porting ============

                                                pack = hand_list.get("shown", [])

                                                # test for validity

                                                (
                                                    validity,
                                                    unrestricted_chi_peng,
                                                    unrestricted_he,
                                                    win_tile_list,
                                                ) = src_py.rule.calc_fan_with_PyMahJongGB_quick(
                                                    tiles,
                                                    planned_tiles,
                                                    pack,
                                                    seat_wind,
                                                    prevalentWind,
                                                    target_fan_val,
                                                    disable_juezhang,
                                                )

                                                # append pack to planned_tiles and tiles
                                                # for entry in hand_list.get("shown", []):
                                                #     # tiles
                                                #     for _key in entry:
                                                #         if _key not in tiles:
                                                #             tiles[_key] = entry[_key]
                                                #         else:
                                                #             tiles[_key] += entry[_key]
                                                # for formed_comp in hand_list.get(
                                                #     "shown", []
                                                # ):
                                                #     planned_tiles.append(formed_comp)

                                                if validity == False:
                                                    continue
                                                if (
                                                    validity == True
                                                    and unrestricted_chi_peng == True
                                                    and unrestricted_he == True
                                                ):
                                                    count_available += 1
                                                    unrestricted_list.append(
                                                        [
                                                            dist,
                                                            tiles,
                                                            availability,
                                                            win_tile_list,
                                                            planned_tiles,
                                                        ]
                                                    )
                                                    if (
                                                        dist < unrestricted_min_dist
                                                        or dist == unrestricted_min_dist
                                                        and availability
                                                        > unrestricted_max_avail
                                                    ):
                                                        unrestricted_max_avail = (
                                                            availability
                                                        )
                                                        unrestricted_min_dist = dist
                                                        unrestricted_id = (
                                                            len(unrestricted_list) - 1
                                                        )

                                                if (
                                                    validity == True
                                                    and unrestricted_chi_peng == True
                                                    and unrestricted_he == False
                                                ):
                                                    count_available += 1
                                                    restricted_lv1_list.append(
                                                        [
                                                            dist,
                                                            tiles,
                                                            availability,
                                                            win_tile_list,
                                                            planned_tiles,
                                                        ]
                                                    )
                                                    if (
                                                        dist < restricted_lv1_min_dist
                                                        or dist
                                                        == restricted_lv1_min_dist
                                                        and availability
                                                        > restricted_lv1_max_avail
                                                    ):
                                                        restricted_lv1_max_avail = (
                                                            availability
                                                        )
                                                        restricted_lv1_min_dist = dist
                                                        restricted_lv1_id = (
                                                            len(restricted_lv1_list) - 1
                                                        )
                                                if (
                                                    validity == True
                                                    and unrestricted_chi_peng == False
                                                    and unrestricted_he == True
                                                ):
                                                    count_available += 1
                                                    restricted_lv2_list.append(
                                                        [
                                                            dist,
                                                            tiles,
                                                            availability,
                                                            win_tile_list,
                                                            planned_tiles,
                                                        ]
                                                    )
                                                    if (
                                                        dist < restricted_lv2_min_dist
                                                        or dist
                                                        == restricted_lv2_min_dist
                                                        and availability
                                                        > restricted_lv2_max_avail
                                                    ):
                                                        restricted_lv2_max_avail = (
                                                            availability
                                                        )
                                                        restricted_lv2_min_dist = dist
                                                        restricted_lv2_id = (
                                                            len(restricted_lv2_list) - 1
                                                        )
                                                if (
                                                    validity == True
                                                    and unrestricted_chi_peng == False
                                                    and unrestricted_he == False
                                                ):
                                                    count_available += 1
                                                    restricted_lv3_list.append(
                                                        [
                                                            dist,
                                                            tiles,
                                                            availability,
                                                            win_tile_list,
                                                            planned_tiles,
                                                        ]
                                                    )
                                                    if (
                                                        dist < restricted_lv3_min_dist
                                                        or dist
                                                        == restricted_lv3_min_dist
                                                        and availability
                                                        > restricted_lv3_max_avail
                                                    ):
                                                        restricted_lv3_max_avail = (
                                                            availability
                                                        )
                                                        restricted_lv3_min_dist = dist
                                                        restricted_lv3_id = (
                                                            len(restricted_lv3_list) - 1
                                                        )

        if count_available >= result_threshold:
            break

    # special combinations
    hand_list_cp = copy.deepcopy(hand_list)
    (
        dist,
        tiles,
        availability,
        win_tile_list,
        planned_tiles,
    ) = src_py.special.bukao(hand_list_cp, tile_list)
    if dist < max_dist:
        restricted_lv2_list.append(
            [
                dist,
                tiles,
                availability,
                win_tile_list,
                planned_tiles,
            ]
        )
        if (
            dist < restricted_lv2_min_dist
            or dist == restricted_lv2_min_dist
            and availability > restricted_lv2_max_avail
        ):
            restricted_lv2_max_avail = availability
            restricted_lv2_min_dist = dist
            restricted_lv2_id = len(restricted_lv2_list) - 1
    hand_list_cp = copy.deepcopy(hand_list)
    (
        dist,
        tiles,
        availability,
        win_tile_list,
        planned_tiles,
    ) = src_py.special.heptapairs(hand_list_cp, tile_list)
    if dist < max_dist:
        restricted_lv2_list.append(
            [
                dist,
                tiles,
                availability,
                win_tile_list,
                planned_tiles,
            ]
        )
        if (
            dist < restricted_lv2_min_dist
            or dist == restricted_lv2_min_dist
            and availability > restricted_lv2_max_avail
        ):
            restricted_lv2_max_avail = availability
            restricted_lv2_min_dist = dist
            restricted_lv2_id = len(restricted_lv2_list) - 1
    hand_list_cp = copy.deepcopy(hand_list)
    (
        dist,
        tiles,
        availability,
        win_tile_list,
        planned_tiles,
    ) = src_py.special.multicolored_dragon(hand_list_cp, tile_list)
    if dist < max_dist:
        restricted_lv2_list.append(
            [
                dist,
                tiles,
                availability,
                win_tile_list,
                planned_tiles,
            ]
        )
        if (
            dist < restricted_lv2_min_dist
            or dist == restricted_lv2_min_dist
            and availability > restricted_lv2_max_avail
        ):
            restricted_lv2_max_avail = availability
            restricted_lv2_min_dist = dist
            restricted_lv2_id = len(restricted_lv2_list) - 1
    hand_list_cp = copy.deepcopy(hand_list)
    (
        dist,
        tiles,
        availability,
        win_tile_list,
        planned_tiles,
    ) = src_py.special.thirteen_yao(hand_list_cp, tile_list)
    if dist < max_dist:
        restricted_lv2_list.append(
            [
                dist,
                tiles,
                availability,
                win_tile_list,
                planned_tiles,
            ]
        )
        if (
            dist < restricted_lv2_min_dist
            or dist == restricted_lv2_min_dist
            and availability > restricted_lv2_max_avail
        ):
            restricted_lv2_max_avail = availability
            restricted_lv2_min_dist = dist
            restricted_lv2_id = len(restricted_lv2_list) - 1

    # # append the index of shortest path combination
    unrestricted_list.append(unrestricted_id)
    restricted_lv1_list.append(restricted_lv1_id)
    restricted_lv2_list.append(restricted_lv2_id)
    restricted_lv3_list.append(restricted_lv3_id)

    return (
        unrestricted_list,
        restricted_lv1_list,
        restricted_lv2_list,
        restricted_lv3_list,
    )
