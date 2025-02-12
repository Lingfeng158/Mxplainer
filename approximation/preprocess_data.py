import os
from collections import defaultdict
import numpy as np
import feature
import copy
import rule
from multiprocessing import Pool
from fanCalcLib import formMinComb_c


## Remove BQR fan, if other fans already met 8-fan threshold
# preprocess_data_revised6 from sl_v6


def default_zero():
    return 0


def default_one():
    return 1


def default_list():
    return []


def default_three_zeros():
    return [0, 0, 0]


def default_seven_zeros():
    return [0, 0, 0, 0, 0, 0, 0]


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


def return_default_filler():
    # Make default filler data
    tile_base_filler = default_seven_zeros()
    tile_base_filler_list = [tile_base_filler for _ in range(34)]
    fan_base_filler_list = [0 for _ in range(80)]
    missing_filler_list = [0 for _ in range(34)]
    coeff_count_filler_list = (0, 0)

    package_filler = (
        tile_base_filler_list,
        fan_base_filler_list,
        missing_filler_list,
        coeff_count_filler_list,
        0,
    )
    return package_filler


def return_default_filler_list():
    package_filler = return_default_filler()
    package_filler_list = [package_filler for _ in range(64)]
    return package_filler_list


def find_redundant_tile(hand_list, selected_tiles):
    redundant_tile = defaultdict(default_zero)
    for t in hand_list:
        if t not in selected_tiles:
            selected_tiles[t] = 0
        if hand_list[t] - selected_tiles[t] > 0:
            redundant_tile[t] += 1
    return redundant_tile


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
    for_QQR: 针对全求人牌型
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
                    dist_dict[dist].append(copy.deepcopy(entry))
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


def chi_peng_utility_differentiable(new_tile, test_chi):
    """
    return possible combinations of chi/peng, (action, midtile(for chi), tile-combination)
    """
    new_tile_type = new_tile[0]
    new_tile_rank = int(new_tile[1])
    test_case_list = []
    if test_chi and new_tile_type in ["B", "W", "T"]:
        for low_rank in range(new_tile_rank - 2, new_tile_rank + 1):
            test_case = {
                new_tile_type + str(low_rank): 1,
                new_tile_type + str(low_rank + 1): 1,
                new_tile_type + str(low_rank + 2): 1,
            }
            if low_rank < 1 or low_rank > 7:
                test_case_list.append(["ILLEGAL", None, {}])
            else:
                test_case_list.append(
                    ["CHI", new_tile_type + str(low_rank + 1), test_case]
                )
    else:
        test_case_list.append(["ILLEGAL", None, {}])
        test_case_list.append(["ILLEGAL", None, {}])
        test_case_list.append(["ILLEGAL", None, {}])
    test_case_list.append(["PENG", new_tile, {new_tile: 3}])
    return test_case_list


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


def craft_one_hot_action_encoding(action, chi_mid_tile, tests_cases):
    """
    Craft one-hot version of [pass/keep-same, chi-1, chi-2, chi-3, peng]
    """
    if action == "Peng":
        return [0, 0, 0, 0, 1]
    elif action == "Chi" and chi_mid_tile == tests_cases[0][1]:
        return [0, 1, 0, 0, 0]
    elif action == "Chi" and chi_mid_tile == tests_cases[1][1]:
        return [0, 0, 1, 0, 0]
    elif action == "Chi" and chi_mid_tile == tests_cases[2][1]:
        return [0, 0, 0, 1, 0]
    else:
        return [1, 0, 0, 0, 0]


def calculate_target_probability_differentiable_prep(
    list_element, hand_list, tile_list, remaining_tileWall=50, opponent_hidden_tiles=39
):
    """
    same functionality as "calculate_target_probability", but formated friendly for supervised learning
    probability = sum(each tile * (some param) * (some param)) * some param
    return probability, redundant_tile
    """
    target_fan_list = list_element[5]
    count_trio = 0
    count_straight = 0
    tile_base = defaultdict(default_seven_zeros)
    fan_base = defaultdict(default_zero)

    # CAN BE processed seperatedly
    for target_fan in target_fan_list:
        fan_base[target_fan] = 1

    # find tiles in need, number in need
    redundant_tile = {
        "CHI": defaultdict(default_zero),
        "PENG": defaultdict(default_zero),
        "OTHER": defaultdict(default_zero),
    }  # 管理当前 list
    # TODO: redundant_tile, desired_tile_list calculate seperately, no intervention
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

        if ele_kind == "TRIO":
            count_trio += 1
        if ele_kind == "STRAIGHT":
            count_straight += 1

        for k in ele:
            if ele[k] != 0:
                if sum(ele.values()) == 1:
                    if ele_kind == "TRIO":
                        redundant_tile["PENG"][k] += ele[k]
                    elif ele_kind == "STRAIGHT":
                        redundant_tile["CHI"][k] += ele[k]
                    else:
                        redundant_tile["OTHER"][k] += ele[k]
                else:
                    redundant_tile["OTHER"][k] += ele[k]

    tile_dict = defaultdict(
        default_three_zeros
    )  # total number of tiles needed, can peng, no need self-draw (0: self_draw only, 1: can chi/peng)
    for dict_key in ["CHI", "PENG", "OTHER"]:
        for k in redundant_tile[dict_key]:
            k_count_need = int(redundant_tile[dict_key][k])
            tile_dict[k][0] += k_count_need
            if dict_key == "PENG":
                tile_dict[k][1] = 1
                tile_dict[k][2] = 1
            if dict_key == "CHI":
                tile_dict[k][2] = 1

    for k in tile_dict:
        total_k_need, can_peng, able_chi_peng = tile_dict[k]
        self_draw_prob = 1

        if tile_list[k] < total_k_need:
            return (
                defaultdict(default_seven_zeros),
                defaultdict(default_zero),
                defaultdict(default_zero),
                (0, 0),
            )

        for i in range(total_k_need):
            self_draw_prob *= (tile_list[k] - i) / (remaining_tileWall - i)

        chi_peng_base = tile_list[k]
        hidden_count = opponent_hidden_tiles
        chi_peng_prob = 0
        chi_peng_switch = 0
        if list_element[-1] in [0, 1]:
            # list 0 and 1 are allowed to chi and peng
            chi_peng_switch = 1
            chi_peng_prob = 1  # * held_prob[k] * discard_prob[k]
            if can_peng != 0:
                # peng: from 3 opponents
                chi_peng_prob *= 2
            if able_chi_peng == 0:
                # duo: not allowed to chi or peng
                chi_peng_prob *= 0
        # print(k, dict_key, self_draw_prob, chi_peng_prob)
        # update tile_base with tile k's prob

        ## TO BE CONTINUED: Make a more reasonable construction
        tile_base[k][0] += self_draw_prob
        tile_base[k][1] += total_k_need
        tile_base[k][2] += chi_peng_prob
        tile_base[k][3] += chi_peng_base
        tile_base[k][4] += hidden_count
        tile_base[k][5] = chi_peng_switch
        tile_base[k][6] = 0

    redundant_tile_list = find_redundant_tile(hand_list, list_element[1])

    return tile_base, fan_base, redundant_tile_list, (count_trio, count_straight)


def summarize_data_prep(
    hand,
    pack,
    tile_list,
    seat_wind,
    prevailing_wind,
    disable_juezhang,
    remaining_tileWall,
    opponent_hidden_tiles,
    target_fan_val=8,
):
    chi_peng_count_remain = 4
    already_chi_peng = False
    for ent in pack:
        if "AnGang" not in ent.keys():
            already_chi_peng = True
            chi_peng_count_remain -= 1
    # total_tile_count = 0
    # total_tile_count += sum(hand.values())
    # for ent in pack:
    #     total_tile_count += sum(ent.values())

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
        hand,
        pack,
        tile_list,
        seat_wind,
        prevailing_wind,
        80,
        8,
        target_fan_val,
        True,
        disable_juezhang,
    )

    list1.append(list1id)
    list2.append(list2id)
    list3.append(list3id)
    list4.append(list4id)
    sorted_list = sortation(list1, list2, list3, list4, already_chi_peng)
    if len(sorted_list) > 56:
        segmented_list_8_fan = sorted_list[:56]
    else:
        segmented_list_8_fan = sorted_list
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
        hand,
        pack,
        tile_list,
        seat_wind,
        prevailing_wind,
        16,
        5,
        2,
        False,
        disable_juezhang,
    )

    list1.append(list1id)
    list2.append(list2id)
    list3.append(list3id)
    list4.append(list4id)
    sorted_list = sortation(list1, list2, list3, list4, already_chi_peng, True)
    if len(sorted_list) > 8:
        segmented_list_2_fan = sorted_list[:8]
    else:
        segmented_list_2_fan = sorted_list

    segmented_list = segmented_list_8_fan + segmented_list_2_fan

    # sample prep
    data_prep_collector = []
    for list_element in segmented_list:
        (
            tile_base,
            fan_base,
            redundant_tile_list,
            coeff_count,
        ) = calculate_target_probability_differentiable_prep(
            list_element, hand, tile_list, remaining_tileWall, opponent_hidden_tiles
        )
        if (
            len(redundant_tile_list) == 0
            and list_element[-1] % 2 == 1
            and len(list_element[3]) != 0
        ):
            return None
        tile_prep = convert_data_representation(
            tile_base, tile_list_raw, [0, 0, 0, 0, 0, 0, 1]
        )
        fan_prep = convert_data_representation(fan_base, fan_list_raw, 0)
        missing_tile_prep = convert_data_representation(
            redundant_tile_list, tile_list_raw, 0
        )
        if "全求人" in list_element[5]:
            data_prep_collector.append(
                (
                    tile_prep,
                    fan_prep,
                    missing_tile_prep,
                    coeff_count,
                    chi_peng_count_remain,
                )
            )
        else:
            data_prep_collector.append(
                (tile_prep, fan_prep, missing_tile_prep, coeff_count, 0)
            )
    for _ in range(64 - len(data_prep_collector)):
        data_prep_collector.append(return_default_filler())
    return data_prep_collector


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


def generate_feature_from_meta(remaining_tile_count, opponent_held_count):
    normalized_remain = remaining_tile_count / 100.0
    normalized_concealed = opponent_held_count / 40.0
    feature_based_on_meta = [
        normalized_remain,
        normalized_remain**2,
        1 - normalized_remain,
        (1 - normalized_remain) ** 2,
        normalized_concealed,
        normalized_concealed**2,
        1 - normalized_concealed,
        (1 - normalized_concealed) ** 2,
    ]
    return feature_based_on_meta


def meta_split(meta):
    """
    Split Meta into four
    Further split till_wall into tile 34* (-2, -1, 0, 1, 2)
    """
    (case_indicator, remaining_tile_count, opponent_held_count, tile_wall_enc) = meta
    tile_wall_tensor = decode_tile_wall(tile_wall_enc)
    # feature_tensor = generate_feature_from_meta(
    #     remaining_tile_count, opponent_held_count
    # )

    return (
        case_indicator,
        [remaining_tile_count, opponent_held_count],
        tile_wall_tensor,
    )


def prepare_data(path_to_data, data_name, data_dst):
    prep_filler_list = return_default_filler_list()
    discard_history_dict_list_adv = [
        defaultdict(default_zero) for _ in range(4)
    ]  # discard history for 4 players
    discard_history_dict_list_slow = [
        defaultdict(default_zero) for _ in range(4)
    ]  # discard history for 4 players, for offset == 0, postpone update
    result_prep_gatherer = []
    truth_label_gatherer = []
    verification_prep_gatherer = []
    verification_label_gatherer = []
    data_info_gatherer = []
    discard_history_gatherer = (
        []
    )  # contains player_id's +1, +2, +3 offsets' play history
    pack_info_gatherer = []
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
    path_to_dst = os.path.join(data_dst, data_name)
    for line_id in range(5, len(botzone_log) - 2):
        line = botzone_log[line_id]
        line_next = botzone_log[line_id + 1]
        keywords = line.split()
        keywords_next = line_next.split()
        all_player_pack = pack_log[line_id]
        discard_history_dict_list_slow = copy.deepcopy(discard_history_dict_list_adv)
        if len(keywords) == 4 and keywords[2] == "Play":
            # Note: discard history inconsistent when player_id_offset = 0, or when ejecting tile
            # inconsistent entry: self, or player_id_offset = 0
            discard_history_dict_list_adv[int(keywords[1])][keywords[3]] += 1

        if keywords[0] != "Player":
            continue
        if keywords[2] == "Deal" or keywords[2] == "Draw":
            continue
        if keywords[2] == "Play" or keywords[2] == "BuGang" or keywords[2] == "AnGang":
            for player_id_offset in range(4):
                player_id = (int(keywords[1]) + player_id_offset) % 4
                # self_hand & pack are only used for offset == 0
                self_hand = handWall_log[line_id - 1][player_id].copy()
                self_pack = pack_log[line_id - 1][player_id]
                # prev_hand, hand, etc. are used for offset == 1,2,3
                prev_hand = handWall_log[line_id][player_id].copy()
                played_tile = keywords[3]
                hand = prev_hand.copy()
                pack = pack_log[line_id][player_id]
                hand[played_tile] += 1
                tile_list = tileWall_log[line_id][player_id]
                tile_list_encoded = convert_data_representation(
                    tile_list, tile_list_raw, 0
                )
                seat_wind = player_id
                opponent_hidden_tiles = 0
                for other_id in range(4):
                    if other_id != player_id:
                        opponent_hidden_tiles += sum(
                            handWall_log[line_id][other_id].values()
                        )
                remaining_tileWall = 0
                for k in tile_list:
                    remaining_tileWall += tile_list[k]
                if player_id_offset == 0:
                    # test ejection, including play, bugang, and angang
                    # disable first term
                    first_term = 0
                    data_prep_8 = summarize_data_prep(
                        self_hand,
                        self_pack,
                        tile_list,
                        seat_wind,
                        prevailing_wind,
                        False,
                        remaining_tileWall,
                        opponent_hidden_tiles,
                        8,
                    )
                    if data_prep_8 == None:
                        continue
                    data_prep_list = [
                        first_term,
                        remaining_tileWall,
                        opponent_hidden_tiles,
                        tile_list_encoded,
                        data_prep_8,
                    ]
                    # calculate pass/default ejection
                    for _ in range(4):
                        # calculate chi case 1-3, peng
                        data_prep_list.append(prep_filler_list)
                    verification_package = [
                        first_term,
                        self_hand,
                        self_pack,
                        tile_list,
                        played_tile,
                        player_id,
                        prevailing_wind,
                        remaining_tileWall,
                        opponent_hidden_tiles,
                    ]
                    result_prep_gatherer.append(data_prep_list)
                    verification_prep_gatherer.append(verification_package)
                    truth_action = "PASS"
                    truth_tile = keywords[3]
                    one_hot_truth_tile = convert_data_representation(
                        {truth_tile: 1}, tile_list_raw, 0
                    )
                    one_hot_truth_action = [1, 0, 0, 0, 0]
                    truth_label_list = [one_hot_truth_action, one_hot_truth_tile]
                    truth_label_gatherer.append(truth_label_list)
                    verification_label_gatherer.append([truth_action, truth_tile])
                    data_info_gatherer.append([data_name, line_id, player_id])
                    # process other players' discard history and shown pack
                    # should contain other players' histories
                    other_player_history_3 = []
                    shown_pack_encoded_3 = []
                    for offset in range(4):
                        other_player_id = (player_id + offset) % 4
                        play_history = discard_history_dict_list_slow[other_player_id]
                        encoded_play_history = convert_data_representation(
                            play_history, tile_list_raw, 0
                        )
                        other_player_history_3.append(encoded_play_history)
                        shown_pack = all_player_pack[other_player_id]
                        pack_tile_collector = defaultdict(default_zero)
                        for entry in shown_pack:
                            for t in entry:
                                pack_tile_collector[t] += entry[t]
                        shown_pack_encoded = convert_data_representation(
                            pack_tile_collector, tile_list_raw, 0
                        )
                        shown_pack_encoded_3.append(shown_pack_encoded)
                    discard_history_gatherer.append(other_player_history_3)
                    pack_info_gatherer.append(shown_pack_encoded_3)

                elif keywords[2] == "Play":
                    # only if action is play
                    # test chi/peng, if  offset ==1 and there is an available combination
                    first_term = 1
                    test_case_chi = False
                    if player_id_offset == 1:
                        test_case_chi = True
                    # prev_win_prob
                    prev_win_prob_8 = summarize_data_prep(
                        prev_hand,
                        pack,
                        tile_list,
                        seat_wind,
                        prevailing_wind,
                        False,
                        remaining_tileWall,
                        opponent_hidden_tiles,
                        8,
                    )
                    if prev_win_prob_8 == None:
                        continue
                    # chi_peng_cases
                    chi_peng_cases = chi_peng_utility_differentiable(
                        played_tile, test_case_chi
                    )

                    case_prep_list = []
                    case_prep_contains_useful_values = False
                    for case in chi_peng_cases:
                        if case[0] == "ILLEGAL":
                            # filler default for illegal case
                            case_prep_list.append(prep_filler_list)
                        else:
                            prerequisite_satisfied = True
                            for t in case[2]:
                                if hand.get(t, 0) < case[2][t]:
                                    prerequisite_satisfied = False
                            if prerequisite_satisfied == False:
                                # filler default for illegal case
                                case_prep_list.append(prep_filler_list)
                            else:
                                case_prep_contains_useful_values = True
                                hand_cp = copy.deepcopy(hand)
                                for t in case[2]:
                                    hand_cp[t] -= case[2][t]

                                pack_cp = pack.copy()
                                pack_cp = np.append(pack_cp, case[2])
                                chi_peng_prep_8 = summarize_data_prep(
                                    hand_cp,
                                    pack_cp,
                                    tile_list,
                                    seat_wind,
                                    prevailing_wind,
                                    True,
                                    remaining_tileWall,
                                    opponent_hidden_tiles,
                                    8,
                                )
                                if chi_peng_prep_8 == None:
                                    continue
                                case_prep_list.append(chi_peng_prep_8)
                    if len(case_prep_list) != 4:
                        continue
                    if case_prep_contains_useful_values == False:
                        continue
                    else:
                        data_prep_list = [
                            first_term,
                            remaining_tileWall,
                            opponent_hidden_tiles,
                            tile_list_encoded,
                            prev_win_prob_8,
                        ]
                        for case_prep in case_prep_list:
                            data_prep_list.append(case_prep)
                        verification_package = [
                            first_term,
                            hand,
                            pack,
                            tile_list,
                            played_tile,
                            player_id,
                            prevailing_wind,
                            remaining_tileWall,
                            opponent_hidden_tiles,
                        ]
                        verification_prep_gatherer.append(verification_package)
                        result_prep_gatherer.append(data_prep_list)
                        if (
                            len(keywords_next) > 4
                            and (int(keywords_next[6]) == player_id)
                            and (
                                keywords_next[7] == "Chi" or keywords_next[7] == "Peng"
                            )
                        ):
                            truth_action = keywords_next[7]
                            truth_tile = keywords_next[8]
                        elif (
                            keywords_next[2] == "Chi"
                            or keywords_next[2] == "Peng"
                            and int(keywords_next[1]) == player_id
                        ):
                            truth_action = keywords_next[2]
                            truth_tile = keywords_next[3]
                        else:
                            truth_action = "PASS"
                            truth_tile = None
                        one_hot_truth_tile = convert_data_representation(
                            {truth_tile: 1}, tile_list_raw, 0
                        )
                        one_hot_truth_action = craft_one_hot_action_encoding(
                            truth_action, truth_tile, chi_peng_cases
                        )
                        truth_label_list = [one_hot_truth_action, one_hot_truth_tile]
                        truth_label_gatherer.append(truth_label_list)
                        verification_label_gatherer.append([truth_action, truth_tile])
                        data_info_gatherer.append([data_name, line_id, player_id])
                        # process other players' discard history and shown pack
                        # should contain other players' histories
                        other_player_history_3 = []
                        shown_pack_encoded_3 = []
                        for offset in range(4):
                            other_player_id = (player_id + offset) % 4
                            play_history = discard_history_dict_list_adv[
                                other_player_id
                            ]
                            encoded_play_history = convert_data_representation(
                                play_history, tile_list_raw, 0
                            )
                            other_player_history_3.append(encoded_play_history)
                            shown_pack = all_player_pack[other_player_id]
                            pack_tile_collector = defaultdict(default_zero)
                            for entry in shown_pack:
                                for t in entry:
                                    pack_tile_collector[t] += entry[t]
                            shown_pack_encoded = convert_data_representation(
                                pack_tile_collector, tile_list_raw, 0
                            )
                            shown_pack_encoded_3.append(shown_pack_encoded)
                        discard_history_gatherer.append(other_player_history_3)
                        pack_info_gatherer.append(shown_pack_encoded_3)
        else:
            continue

    # Further process meta data
    # Now data_list: [32*[_8_data, _8_min_dist, _8_avg_dist, _4_data, _4_min_dist, _4_avg_dist]]
    meta_list = []
    raw_meta_feature_list = []
    tile_wall_feature_list = []
    data_list = []
    label_action_list = []
    label_tile_list = []

    for ele in result_prep_gatherer:
        meta = ele[:4]
        data = ele[4:]
        (case_indicator, raw_features, tile_wall_features) = meta_split(meta)
        meta_list.append(case_indicator)
        raw_meta_feature_list.append(raw_features)
        tile_wall_feature_list.append(tile_wall_features)
        data_list.append(data)
    for ele in truth_label_gatherer:
        label_action, label_tile = ele
        label_action_list.append(label_action)
        label_tile_list.append(label_tile)

    with open(path_to_dst, "wb") as f:
        np.save(f, np.array(meta_list))
        np.save(f, np.array(raw_meta_feature_list))
        np.save(f, np.array(tile_wall_feature_list))
        np.save(f, np.array(discard_history_gatherer))
        np.save(f, np.array(pack_info_gatherer))
        np.save(f, np.array(data_list, dtype=object))
        np.save(f, np.array(label_action_list))
        np.save(f, np.array(label_tile_list))
        np.save(f, np.array(verification_prep_gatherer, dtype=object))
        np.save(f, np.array(verification_label_gatherer, dtype=object))
        np.save(f, np.array(data_info_gatherer))


if __name__ == "__main__":
    cpuCount = os.cpu_count() - 4
    path = "data"
    dst = "sl_prep_revised6_QQR"
    if not os.path.isdir(dst):
        os.makedirs(dst)
    # file = "10740.npy"
    # prepare_data(path, file, dst)
    file_list = os.listdir(path)
    file_list = sorted(file_list)[:50000]
    pool = Pool(cpuCount)
    for fil in file_list:
        pool.apply_async(
            prepare_data,
            args=(path, fil, dst),
        )
        # print("Processing {}".format(fil))
        # prepare_data(path, fil, dst)
    pool.close()
    pool.join()
    print("Post Processing Done!")
