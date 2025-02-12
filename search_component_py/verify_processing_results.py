from feature import DataFormatting
from collections import defaultdict
import copy
import os
import feature
import rule

# check validity of preprocessing
path = "data/"
# file = "2.npy"
dir_list = os.listdir(path)
err_ct = 0
processed_ct = 0
for file in dir_list:
    processed_ct += 1
    if processed_ct % 1024 == 0:
        print("{} entries done processing".format(processed_ct))
    (
        botzone_log,
        _,
        pack,
        handWall,
        obsWall,
        remaining_tile,
        _,
        winner_id,
        wind,
        fan_sum,
        fan_list,
    ) = feature.load_log(path, file)

    if fan_sum != -1:
        win_tile = botzone_log[-1].split()[3]
        pack = pack[-1][winner_id]
        handWall = handWall[-1][winner_id]
        obsWall = obsWall[-1][0]

        is_4th = obsWall[win_tile] == 0
        is_self = botzone_log[-1].split()[1] == botzone_log[-2].split()[1]
        if is_self:
            is_4th = obsWall[win_tile] == 1
            remaining_tile = remaining_tile[-1][(winner_id + 1) % 4]
        else:
            is_4th = obsWall[win_tile] == 0
            prev_player_id = int(botzone_log[-2].split()[1])
            remaining_tile = remaining_tile[-1][(prev_player_id + 1) % 4]
        is_last = remaining_tile == 0
        # print(file)
        if (
            (
                "Gang" in botzone_log[-3].split()
                or "BuGang" in botzone_log[-3].split()
                or "AnGang" in botzone_log[-3].split()
            )
            and is_self
            # or "Gang" in botzone_log[-1].split()
            or "BuGang" in botzone_log[-2].split()
        ):
            is_kong = True
        else:
            is_kong = False
        fan_sum_calc, fan_list_calc = rule.calc_exact_fan_with_PyMahJongGB(
            pack,
            handWall,
            win_tile,
            is_last,
            is_self,
            is_4th,
            is_kong,
            winner_id,
            wind,
        )
        if fan_sum != fan_sum_calc:
            err_ct += 1
            # print(fan_list, fan_list_calc)
            # print(fan_sum, fan_sum_calc)
            print("ERROR IN File {}".format(file))
print("CHECK DONE, ERROR COUNT: {}!".format(err_ct))


# print(e, f, g, h)
