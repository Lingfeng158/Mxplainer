# Model part
import torch
from torch import nn
import torch.nn.functional as F
from collections import defaultdict
from itertools import chain


def default_zero():
    return 0


# no term2 network
# no chi_peng_prob_coeff
# no self.chi_peng_coeff
# shared weights
# 1 channel cnn
# model_TFx_OB_v6e_mod3 from sl_v6
# linear v2, no QQR

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


def reversed_tile_conversion(list_rep, key_order_list):
    ret_dict = defaultdict(default_zero)
    for i in range(len(key_order_list)):
        ret_dict[key_order_list[i]] += list_rep[i]
    return ret_dict


class MahJongNetBatchedRevised(nn.Module):
    """
    This module computes data_prep ([32*4, 80, 32, 2]) object
    """

    def __init__(self, device="cpu"):
        super(MahJongNetBatchedRevised, self).__init__()
        self.device = device
        self.fan_coeff = nn.Parameter(
            torch.tensor(
                [0.1 + torch.randn(1) * 0.015 for _ in range(80)],
                requires_grad=True,
            )
        )
        self.fan_coeff_multiplier = torch.tensor(
            [1.0 for _ in range(80)], device=device
        )
        self.fan_coeff_multiplier[33] *= 21
        self.tile_coeff = nn.Parameter(
            torch.tensor(
                [1.0 + torch.randn(1) * 0.015 for _ in range(34)], requires_grad=True
            )
        )
        # self.QQR_pack_penalty = nn.Parameter(torch.tensor(0.1, requires_grad=True))

        self.prob_module_throw = TileProbModuleNew().to(device)

    def access_named_params(self):
        return (
            self.fan_coeff.detach().numpy(),
            self.tile_coeff.detach().numpy(),
            # self.QQR_pack_penalty.item(),
        )

    def access_fan(self):
        return self.fan_coeff

    def forward(self, x):
        """
        @data_entry: n * 5 * 32 * (34*4, 80, 34, 2)
        @held_prob: n * 5 * 32 * (34)
        @discard_prob: n * 5 * 32 * (34)
        """

        (meta_feature, tile_wall_feature, search_matrix) = x

        prob_throw = self.prob_module_throw(meta_feature, tile_wall_feature)

        (
            tile_prep_data,
            fan_prep,
            redundant_tile_prep_data,
            count_prep,
            chi_peng_count_remain_data,
        ) = search_matrix

        redundant_tile_prep = redundant_tile_prep_data.view(-1, 320, 34)
        chi_peng_count_remain = chi_peng_count_remain_data.view(-1, 320)
        tile_prep = tile_prep_data.view(-1, 320, 34, 7)
        tile_self_draw_baseline = tile_prep[:, :, :, 0]
        tile_need_count = tile_prep[:, :, :, 1]
        tile_cp_coeff = tile_prep[:, :, :, 2]
        tile_cp_base = tile_prep[:, :, :, 3]
        tile_hidden_ct = tile_prep[:, :, :, 4]
        tile_cp_switch = tile_prep[:, :, :, 5]
        tile_null_filler = tile_prep[:, :, :, 6]

        fan_prep_prime = 1 - fan_prep

        # Throw operation
        # fan preference
        fan_pref = fan_prep * self.fan_coeff * self.fan_coeff_multiplier
        multiplied_fan_pref_throw = torch.sum(fan_pref, dim=-1).view(
            -1, 320
        )  # n * 5 * 64

        # tile probability (throw)
        term1_throw = (
            tile_self_draw_baseline * prob_throw.unsqueeze(1) ** tile_need_count
        )
        term2_throw = tile_cp_base * tile_cp_coeff / tile_hidden_ct  #
        term2_throw_zeroed = torch.nan_to_num(term2_throw)
        term3_throw = tile_null_filler
        terms_summed_throw = (
            term1_throw + term3_throw + term2_throw_zeroed * tile_cp_switch
        )

        multiplied_sum_throw = (
            torch.prod(terms_summed_throw, -1) * 100 * multiplied_fan_pref_throw
        )
        final_prob_throw = (
            multiplied_sum_throw  # * self.QQR_pack_penalty**chi_peng_count_remain
        )

        # distinguish by redundant tile (Throw)
        redundant_tile_prob_throw = redundant_tile_prep * final_prob_throw.unsqueeze(-1)
        # missing_tile_prob_throw += tile_pref_from_fan

        # combine 64 sub entries
        # win-rate transfered to tile, judge tile by tile-bounded win-rate
        redundant_tile_prob_throw = redundant_tile_prob_throw.view(-1, 5, 64, 34)
        tile_bound_winrate_throw = redundant_tile_prob_throw.sum(dim=-2)

        # final tile-bounded prob
        weighted_tile_prob = (tile_bound_winrate_throw * self.tile_coeff)[:, 0]

        # combine 64 sub entries
        # reshape
        # final_prob: win rate for each 64 entries
        final_prob_ops = final_prob_throw.view(-1, 5, 64)
        # win-rate transfered to tile, judge tile by tile-bounded win-rate
        missing_tile_prob_ops = redundant_tile_prob_throw
        tile_bound_winrate_ops = missing_tile_prob_ops.sum(dim=2)
        final_prob_ops = final_prob_ops.sum(dim=-1)

        # produce result
        # final tile-bounded prob
        max_tile_prob_for_action, e = torch.max(tile_bound_winrate_ops, dim=2)
        max_tile_prob_for_action[:, 0] = final_prob_ops[:, 0]
        # max_tile_prob_for_action *= self.chi_peng_coeff

        return (
            max_tile_prob_for_action,
            weighted_tile_prob,
        )
