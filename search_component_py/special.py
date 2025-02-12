import sys
import os

sys.path.append("/data")
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import numpy as np
import src_py.rule
import MahjongGB
import copy


def thirteen_yao(hand_list, tile_list):
    """
    14 张牌, 看距离13幺的胡牌距离
    """
    canonical_hand_list = src_py.rule.from_custom_to_canonical_encoding(hand_list)
    hand_list_cp = hand_list.copy()
    selected_tile = {}
    availability = 0
    if len(canonical_hand_list) < 13:
        # 已有吃碰
        return (
            13,
            {},
            0,
            [],
            [],
        )
    # fmt: off
    thirteen_yao_tile_reference_list = ['W1', 'W9', 'T1', 'T9', 'B1', 'B9', 'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3']
    thirteen_yao_tile_list = ['W1', 'W9', 'T1', 'T9', 'B1', 'B9', 'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3']
    # fmt: on
    # each at least one
    for tile in hand_list:
        if tile in thirteen_yao_tile_list:
            hand_list_cp[tile] -= 1
            thirteen_yao_tile_list.remove(tile)
            if tile not in selected_tile:
                selected_tile[tile] = 1
            else:
                selected_tile[tile] += 1
            # clean up if tile count reaches 0
            if hand_list_cp[tile] == 0:
                del hand_list_cp[tile]

    # one of the 13 needs another
    additional_tile = None
    second_available = False
    for tile in hand_list_cp:
        if tile in thirteen_yao_tile_reference_list:
            second_available = True
            hand_list_cp[tile] -= 1
            if tile not in selected_tile:
                selected_tile[tile] = 1
            else:
                selected_tile[tile] += 1
            additional_tile = tile
            break

    max_avail = 0
    max_avail_is_in_hand = False
    max_avail_tile = None

    for tile in thirteen_yao_tile_reference_list:
        single_tile_avail = tile_list[tile]
        is_in_hand = tile not in thirteen_yao_tile_list
        availability += 0 if is_in_hand else single_tile_avail
        effective_avail = (
            single_tile_avail
            if (is_in_hand or second_available)
            else single_tile_avail / 2.0
        )
        if single_tile_avail == 0 and not is_in_hand:
            return (
                13,
                {},
                0,
                [],
                [],
            )
        if effective_avail > max_avail and not second_available:
            max_avail = effective_avail
            max_avail_tile = tile
            max_avail_is_in_hand = is_in_hand
    if not second_available:
        if max_avail_is_in_hand:
            availability += max_avail

    additional_tile = additional_tile if second_available else max_avail_tile
    # dist = (
    #     13 - len(thirteen_yao_tile_list)
    #     if second_available == True
    #     else 14 - len(thirteen_yao_tile_list)
    # )

    thirteen_yao_tile_reference_list.append(additional_tile)
    if second_available == True:
        dist = len(thirteen_yao_tile_list)
        return (
            dist,
            selected_tile,
            availability,
            [],
            [
                src_py.rule.from_canonical_to_custom_encoding(
                    thirteen_yao_tile_reference_list
                )
            ],
        )
    else:
        # second_available == False
        dist = len(thirteen_yao_tile_list) + 1
        return (
            dist,
            selected_tile,
            availability,
            [],
            [
                src_py.rule.from_canonical_to_custom_encoding(
                    thirteen_yao_tile_reference_list
                )
            ],
        )


def bukao(hand_list, tile_list):
    """
    14张牌, 到全不靠、七星不靠距离
    """
    canonical_hand_list = src_py.rule.from_custom_to_canonical_encoding(hand_list)
    len_pack = len(hand_list.get("shown", []))
    if len(canonical_hand_list) < 13 or len_pack != 0:
        # 已有吃碰
        return (
            13,
            {},
            0,
            [],
            [],
        )
    dragon_list = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    dragon_type_reference = ["B", "W", "T"]
    dragon_type = ["B", "W", "T"]
    min_dragon_dist = 13
    min_dragon_req = []
    min_dragon_hand_remain = []
    min_dragon_sel = []
    min_dragon_targ = []
    max_dragon_avail = 0
    for type1 in dragon_type_reference:

        dragon_type_cp = dragon_type.copy()
        dragon_type_cp.remove(type1)
        for type2 in dragon_type_cp:
            if type2 == dragon_type_cp[0]:
                type3 = dragon_type_cp[1]
            else:
                type3 = dragon_type_cp[0]
            # make multicolored dragon
            component1 = dragon_list[0]
            component2 = dragon_list[1]
            component3 = dragon_list[2]
            multicolored_dragon_tile = []
            for rank in component1:
                multicolored_dragon_tile.append(type1 + str(rank))
            for rank in component2:
                multicolored_dragon_tile.append(type2 + str(rank))
            for rank in component3:
                multicolored_dragon_tile.append(type3 + str(rank))

            bukao_tile = multicolored_dragon_tile + [
                "F1",
                "F2",
                "F3",
                "F4",
                "J1",
                "J2",
                "J3",
            ]
            bukao_tile_reference = bukao_tile.copy()

            # try formed bukao and record distance
            can_hand_list_cp = canonical_hand_list.copy()
            sel_tile_list = []
            for tile in bukao_tile_reference:
                if tile in can_hand_list_cp:
                    bukao_tile.remove(tile)
                    can_hand_list_cp.remove(tile)
                    sel_tile_list.append(tile)

            dragon_dist = len(bukao_tile)
            dragon_avail = 0
            zeroed_tiles = 0
            for tile in bukao_tile:
                avail = tile_list[tile]
                dragon_avail += avail
                if avail == 0:
                    zeroed_tiles += 1
            if zeroed_tiles > 2:
                continue
            if dragon_dist < min_dragon_dist:
                min_dragon_dist = dragon_dist
                min_dragon_req = bukao_tile
                min_dragon_hand_remain = can_hand_list_cp
                min_dragon_sel = sel_tile_list
                max_dragon_avail = dragon_avail
                min_dragon_targ = [
                    src_py.rule.from_canonical_to_custom_encoding(bukao_tile_reference)
                ]

    target = min_dragon_targ
    # target = [rule.from_canonical_to_custom_encoding(target)]
    return (
        min_dragon_dist - 2,
        src_py.rule.from_canonical_to_custom_encoding(min_dragon_sel),
        max_dragon_avail,
        [],
        target,
    )


def heptapairs(hand_list, tile_list):
    """
    七小对
    """
    selected_tile = {}
    target_tile = {}
    formed_pairs = 0
    for tile in hand_list:
        if tile != "shown":
            if hand_list[tile] == 4:
                hand_list[tile] -= 4
                selected_tile[tile] = 4
                target_tile[tile] = 4
                formed_pairs += 2
            if hand_list[tile] >= 2:
                hand_list[tile] -= 2
                selected_tile[tile] = 2
                target_tile[tile] = 2
                formed_pairs += 1
    canonical_hand_list = src_py.rule.from_custom_to_canonical_encoding(hand_list)
    if len(canonical_hand_list) < 13 - formed_pairs * 2:
        # 已有吃碰
        # fmt: off
        return (13,{},0,[], [],)
        # fmt: on
    tile_type_list = ["B", "W", "T", "X"]
    final_dist = 0
    final_avail = 0
    dist_arr = np.array([])
    avail_arr = np.array([])
    index_arr = np.array(range(34))
    ranked_index_list = []
    for one_type in tile_type_list:
        dist_to_duo, avail_tile_to_duo = src_py.rule.one_type_dist_to_duo(
            one_type, hand_list, tile_list
        )
        dist_arr = np.concatenate((dist_arr, dist_to_duo))
        avail_arr = np.concatenate((avail_arr, avail_tile_to_duo))

    dist_arr = dist_arr.astype(int)
    avail_arr = avail_arr.astype(int)
    dist0_idx = index_arr[dist_arr == 0]
    ranked_index_list = np.concatenate((ranked_index_list, dist0_idx))
    dist1_avail3_idx = index_arr[(dist_arr == 1) & (avail_arr == 3)]
    ranked_index_list = np.concatenate((ranked_index_list, dist1_avail3_idx))
    dist1_avail2_idx = index_arr[(dist_arr == 1) & (avail_arr == 2)]
    ranked_index_list = np.concatenate((ranked_index_list, dist1_avail2_idx))
    dist1_avail1_idx = index_arr[(dist_arr == 1) & (avail_arr == 1)]
    ranked_index_list = np.concatenate((ranked_index_list, dist1_avail1_idx))
    dist2_avail4_idx = index_arr[(dist_arr == 2) & (avail_arr == 4)]
    ranked_index_list = np.concatenate((ranked_index_list, dist2_avail4_idx))
    dist2_avail3_idx = index_arr[(dist_arr == 2) & (avail_arr == 3)]
    ranked_index_list = np.concatenate((ranked_index_list, dist2_avail3_idx))
    dist2_avail2_idx = index_arr[(dist_arr == 2) & (avail_arr == 2)]
    ranked_index_list = np.concatenate((ranked_index_list, dist2_avail2_idx))
    ranked_index_list = ranked_index_list.astype(int)
    if len(ranked_index_list) > 6 - formed_pairs:
        for i in range(7 - formed_pairs):
            index = ranked_index_list[i]
            dist = dist_arr[index]
            avail = avail_arr[index]
            tile_type = tile_type_list[index // 9]
            tile_rank = index - 9 * (index // 9)
            if tile_type == "X":
                if tile_rank < 4:
                    tile_type = "F"
                else:
                    tile_type = "J"
                    tile_rank = tile_rank - 4
            tile_rank += 1
            tile = tile_type + str(tile_rank)
            if dist == 0:
                final_dist += dist
                selected_tile[tile] = 2
                target_tile[tile] = 2
            elif dist >= 1 and avail >= dist:
                final_dist += dist
                selected_tile[tile] = 2 - dist
                target_tile[tile] = 2
                final_avail += avail_arr[index]
            else:
                # fmt: off
                return (13,{},0,[], [],)
                # fmt: on
    else:
        # fmt: off
        return (13,{},0,[], [],)
        # fmt: on
    return final_dist, selected_tile, final_avail, [], [target_tile]


def multicolored_dragon(hand_list, tile_list):
    """
    花龙， 优先考虑花龙, 9+3+2
    """
    canonical_hand_list = src_py.rule.from_custom_to_canonical_encoding(hand_list)
    if len(canonical_hand_list) + len(hand_list["shown"]) >= 13:
        dragon_list = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
        dragon_type_reference = ["B", "W", "T"]
        dragon_type = ["B", "W", "T"]
        min_dragon_dist = 9
        min_dragon_req = []
        min_dragon_hand_remain = []
        min_dragon_sel = []
        min_dragon_targ = []
        max_dragon_avail = 0
        for type1 in dragon_type_reference:

            dragon_type_cp = dragon_type.copy()
            dragon_type_cp.remove(type1)
            for type2 in dragon_type_cp:
                if type2 == dragon_type_cp[0]:
                    type3 = dragon_type_cp[1]
                else:
                    type3 = dragon_type_cp[0]
                # make multicolored dragon
                component1 = dragon_list[0]
                component2 = dragon_list[1]
                component3 = dragon_list[2]
                multicolored_dragon_tile = []
                for rank in component1:
                    multicolored_dragon_tile.append(type1 + str(rank))
                for rank in component2:
                    multicolored_dragon_tile.append(type2 + str(rank))
                for rank in component3:
                    multicolored_dragon_tile.append(type3 + str(rank))

                multicolored_dragon_tile_reference = multicolored_dragon_tile.copy()

                # try formed dragon and record distance
                can_hand_list_cp = canonical_hand_list.copy()
                sel_tile_list = []
                for tile in multicolored_dragon_tile_reference:
                    if tile in can_hand_list_cp:
                        multicolored_dragon_tile.remove(tile)
                        can_hand_list_cp.remove(tile)
                        sel_tile_list.append(tile)

                dragon_dist = len(multicolored_dragon_tile)
                dragon_avail = 0
                zero_avail_detection = False
                for tile in multicolored_dragon_tile:
                    avail = tile_list[tile]
                    dragon_avail += avail
                    if avail == 0:
                        zero_avail_detection = True
                if zero_avail_detection:
                    continue
                if dragon_dist < min_dragon_dist:
                    min_dragon_dist = dragon_dist
                    min_dragon_req = multicolored_dragon_tile
                    min_dragon_hand_remain = can_hand_list_cp
                    min_dragon_sel = sel_tile_list
                    max_dragon_avail = dragon_avail
                    min_dragon_targ = [
                        src_py.rule.from_canonical_to_custom_encoding(
                            multicolored_dragon_tile_reference
                        )
                    ]

        # with min dragon information
        # form 3+2
        tile_list_cp = tile_list.copy()
        tile_list_cp = src_py.rule.update_tile_info(
            tile_list_cp, src_py.rule.from_canonical_to_custom_encoding(min_dragon_req)
        )

        if len(hand_list["shown"]) == 1:
            # a triplet is formed already in pack
            min_duo_dist = 2
            min_duo_sel = {}
            min_duo_targ = {}
            max_duo_avail = 0
            tile_type_list = ["B", "W", "T", "X"]
            for tile_type in tile_type_list:
                (duo_dist, duo_comp, duo_avail,) = src_py.rule.one_type_min_dist_to_duo(
                    tile_type,
                    src_py.rule.from_canonical_to_custom_encoding(
                        min_dragon_hand_remain
                    ),
                    tile_list_cp,
                )
                if (
                    duo_dist < min_duo_dist
                    or duo_dist == min_duo_dist
                    and duo_avail > max_duo_avail
                ):
                    min_duo_dist = duo_dist
                    min_duo_sel, min_duo_targ = duo_comp
                    max_duo_avail = duo_avail if min_duo_dist != 0 else 0
                    min_duo_sel = src_py.rule.from_custom_to_canonical_encoding(
                        min_duo_sel
                    )
            ret_tile_targ = min_dragon_targ.copy()
            ret_tile_targ.append(min_duo_targ)
            # ret_tile_targ.append(hand_list["shown"][0])
            return (
                min_dragon_dist + min_duo_dist,
                src_py.rule.from_canonical_to_custom_encoding(
                    min_dragon_sel + min_duo_sel
                ),
                max_dragon_avail + max_duo_avail,
                [],
                ret_tile_targ,
            )
        else:
            # need a triplet and a duo
            min_duo_dist = 2
            min_duo_sel = {}
            min_duo_targ = {}
            max_duo_avail = 0
            min_triplet_dist = 3
            min_triplet_sel = {}
            min_triplet_targ = {}
            max_triplet_avail = 0
            tile_type_list = ["B", "W", "T", "X"]
            for tile_type in tile_type_list:
                (
                    tri_dist,
                    tri_comp,
                    tri_avail,
                ) = src_py.rule.one_type_min_dist_to_any_triplet(
                    tile_type,
                    src_py.rule.from_canonical_to_custom_encoding(
                        min_dragon_hand_remain
                    ),
                    tile_list_cp,
                    tile_type == "X",
                )
                if (
                    tri_dist < min_triplet_dist
                    or tri_dist == min_triplet_dist
                    and tri_avail > max_triplet_avail
                ):
                    min_triplet_dist = tri_dist
                    min_triplet_sel, min_triplet_targ = tri_comp
                    max_triplet_avail = tri_avail if min_triplet_dist != 0 else 0
                    min_triplet_sel = src_py.rule.from_custom_to_canonical_encoding(
                        min_triplet_sel
                    )

            # update tile_list
            tile_list_cp = src_py.rule.update_tile_info(tile_list_cp, min_triplet_targ)
            # update hand_list
            hand_list_remain = min_dragon_hand_remain.copy()
            for sel in min_triplet_sel:
                hand_list_remain.remove(sel)

            for tile_type in tile_type_list:
                (duo_dist, duo_comp, duo_avail,) = src_py.rule.one_type_min_dist_to_duo(
                    tile_type,
                    src_py.rule.from_canonical_to_custom_encoding(hand_list_remain),
                    tile_list_cp,
                )
                if (
                    duo_dist < min_duo_dist
                    or duo_dist == min_duo_dist
                    and duo_avail > max_duo_avail
                ):
                    min_duo_dist = duo_dist
                    min_duo_sel, min_duo_targ = duo_comp
                    max_duo_avail = duo_avail if min_duo_dist != 0 else 0
                    min_duo_sel = src_py.rule.from_custom_to_canonical_encoding(
                        min_duo_sel
                    )
            ret_tile_targ = min_dragon_targ.copy()
            ret_tile_targ.append(min_duo_targ)
            ret_tile_targ.append(min_triplet_targ)
            return (
                min_dragon_dist + min_duo_dist + min_triplet_dist,
                src_py.rule.from_canonical_to_custom_encoding(
                    min_dragon_sel + min_duo_sel + min_triplet_sel
                ),
                max_dragon_avail + max_duo_avail + max_triplet_avail,
                [],
                ret_tile_targ,
            )
    else:
        return (
            13,
            {},
            0,
            [],
            [],
        )
