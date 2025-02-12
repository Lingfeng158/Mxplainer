import numpy as np
import torch
import os
from torch import nn
from collections import defaultdict
import json
from model import MahJongNetBatchedRevised as model

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


def default_zero():
    return 0


def reversed_tile_conversion(list_rep, key_order_list):
    ret_dict = defaultdict(default_zero)
    for i in range(len(key_order_list)):
        ret_dict[key_order_list[i]] += list_rep[i]
    return ret_dict


if __name__ == "__main__":
    # change params here
    log_name = "07_02_20_34_26-model_revA_full"
    device = "cpu"
    net = model(device).to(device)

    # typically leave as is
    logdir = "log/"
    log_suffix = "checkpoint"
    extract_dir = "extracted_model_params"
    weight_list = {
        "top1": "best_acc.pkl",
        "top2": "best_acc_top2.pkl",
        "top3": "best_acc_top3.pkl",
    }
    for weight in weight_list:
        # create path for model parameter extraction
        extract_path = os.path.join(extract_dir, weight)
        if not os.path.isdir(extract_path):
            os.makedirs(extract_path)

        # load model parameter from log
        net.load_state_dict(
            torch.load(
                os.path.join(logdir, log_name, log_suffix, weight_list[weight]),
                map_location=torch.device(device),
            )
        )
        torch.save(
            net.prob_module_throw.state_dict(),
            "{}/{}.pkl".format(extract_path, "net"),
            _use_new_zipfile_serialization=False,
        )
        fan_coeff, tile_coeff, QQR_penalty = net.access_named_params()
        fan_tile_dict = {}
        for fan_id in range(80):
            tile_dict = reversed_tile_conversion(fan_coeff[fan_id], tile_list_raw)
            fan_tile_dict[fan_list_raw[fan_id]] = tile_dict
        tile_dict = reversed_tile_conversion(tile_coeff, tile_list_raw)

        with open("{}/global.json".format(extract_path), "w") as f:
            json.dump(
                {
                    "fan": fan_tile_dict,
                    "tile": tile_dict,
                    "QQR": QQR_penalty,
                },
                f,
            )
