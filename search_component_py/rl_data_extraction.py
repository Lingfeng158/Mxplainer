import numpy as np
import feature
from collections import Counter
import json
from multiprocessing import Pool
import os


def extract_positive_datum(path, name):
    file_name = "{}.npy".format(name)
    output = feature.load_log(path, file_name)
    (
        botzone_log,  # 1
        tileWall_log,  # 2
        pack_log,  # 3
        handWall_log,  # 4
        obsWall_log,  # 5
        remaining_tile_log,  # 6
        botzone_id,  # 7
        winner_id,  # 8
        prevalingWind,  # 9
        fan_sum,  # 10
        score,
        fan_list,  # 11
    ) = output
    winning_tile = (
        list(
            Counter(handWall_log[-1][winner_id]) - Counter(handWall_log[-2][winner_id])
        ),
    )
    return (
        # botzone_log,
        tileWall_log[-2],  # 1
        pack_log[-2],  # 2
        handWall_log[-2],  # 3
        obsWall_log[-2],  # 4
        remaining_tile_log[-2],  # 5
        # botzone_id,
        winning_tile,  # 6
        winner_id,  # 7
        prevalingWind,  # 8
        # fan_sum,
        # score,
        fan_list,  # 9
    )


if __name__ == "__main__":
    path = "../data/"
    data_collection1 = []
    data_collection2 = []
    data_collection3 = []
    data_collection4 = []
    data_collection5 = []
    data_collection6 = []
    data_collection7 = []
    data_collection8 = []
    data_collection9 = []
    # cpuCount = os.cpu_count() - 2
    # pool = Pool(cpuCount)
    # ret_list = []
    # file_list = os.listdir(path)
    for i in range(98209):
        # if i == 121:
        #     continue
        # for fil in file_list:
        d = extract_positive_datum(path, i)
        #     ret = pool.apply_async(
        #         extract_positive_datum,
        #         args=(path, fil),
        #     )
        #     ret_list.append(ret)
        # pool.close()
        # pool.join()
        # for ret in ret_list:
        #     d = ret.get()
        #     if d[6] >= 0 and d[6] <= 3:
        # print(d[2].shape == 4)

        if d[1].shape == (4, 0):
            continue
        if d[6] >= 0 and d[6] <= 3:
            data_collection1.append(d[0])
            data_collection2.append(d[1])
            data_collection3.append(d[2])
            data_collection4.append(d[3])
            data_collection5.append(d[4])
            data_collection6.append(d[5])
            data_collection7.append(d[6])
            data_collection8.append(d[7])
            data_collection9.append(d[8])

    with open("positives.npy", "wb") as f:
        np.save(f, np.array(data_collection1, dtype=object))
        np.save(f, np.array(data_collection2, dtype=object))
        np.save(f, np.array(data_collection3, dtype=object))
        np.save(f, np.array(data_collection4, dtype=object))
        np.save(f, np.array(data_collection5, dtype=object))
        np.save(f, np.array(data_collection6, dtype=object))
        np.save(f, np.array(data_collection7, dtype=object))
        np.save(f, np.array(data_collection8, dtype=object))
        np.save(f, np.array(data_collection9, dtype=object))
