from MahJong_CNN_model2 import MahJongCNNNet2
from MahJong_Attention_model2 import MahJongNet2
from MahJong_Attention_model3 import MahJongNet3
from MahJong_Attention_model4 import MahJongNet4
import torch
from dataset import MahjongGBDataset
import json
from multiprocessing import Pool
import _pickle as cPickle
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, ExponentialLR


def workload(idx_list, start_idx):
    cache = {"obs": [], "mask": [], "act": []}
    for i in idx_list:
        d = np.load("data/processed_data/%d.npz" % (i + start_idx))
        for k in d:
            cache[k].append(d[k])
    return cache


if __name__ == "__main__":
    input_shape = (2, 76, 4, 9)
    d_input = torch.rand(input_shape)
    input = {}
    input["observation"] = d_input
    input["action_mask"] = torch.ones(1, 235)
    n = MahJongNet2("cpu")
    # sd = torch.load("./30.pkl")
    # n.load_state_dict(sd)
    a = n(input)
    print(sys.getsizeof(cPickle.dumps(n.state_dict())))
    # print(a[0])
    # a = [i % 5 for i in range(20)]
    # a = torch.tensor(a)
    # print(a)
    # print(a.view(4, -1))

    # begin = 0
    # end = 0.9
    # num_workers = 4
    # with open("data/count.json") as f:
    #     match_samples = json.load(f)
    # total_matches = len(match_samples)  # total number of matches
    # total_samples = sum(match_samples)  # total number of trainable samples
    # begin = int(begin * total_matches)  # start location by match
    # end = int(end * total_matches)  # end location by match
    # match_samples = match_samples[begin:end]  # select by match
    # matches = len(match_samples)
    # samples = sum(match_samples)
    # t = 0
    # for i in range(matches):  # convert match samples to exclusive prefix sum
    #     a = match_samples[i]  # a: samples in one match
    #     match_samples[i] = t  # update i-th match to exclusive prefix sum
    #     t += a
    # # At this point, match_samples contains exclusive prefix sum of sample matches (sum of total trainable rounds up to)

    # cache = {"obs": [], "mask": [], "act": []}
    # ret_list = []
    # pool = Pool(num_workers)
    # for i in range(num_workers):
    #     ret = pool.apply_async(
    #         workload,
    #         args=(
    #             range(
    #                 int(matches / num_workers * i),
    #                 int(matches / num_workers * (i + 1)),
    #             ),
    #             begin,
    #         ),
    #     )
    #     ret_list.append(ret)
    # pool.close()
    # pool.join()
    # for ret in ret_list:
    #     d = ret.get()
    #     for k in d:
    #         cache[k].extend(d[k])

    # for k in cache:
    #     cache[k] = np.concatenate(cache[k], axis=0)

    # print(cache["obs"].shape, cache["mask"].shape, cache["act"].shape)

    # tDS = MahjongGBDataset(0.93, 1)
    # tloader = DataLoader(dataset=tDS, batch_size=512, shuffle=True, num_workers=8)
    # for i, d in enumerate(tloader):
    #     obs = d[0]
    #     mask = d[1]
    #     act = d[2]
    #     print(obs.shape, mask.shape, act.shape)
    # number_warmup_steps = 10

    # def warmup(current_step: int):
    #     if current_step > number_warmup_steps:
    #         return 1
    #     else:
    #         return 1 / (1.5 ** (float(number_warmup_steps - current_step)))

    # optimizer = torch.optim.Adam(n.parameters(), lr=1e-4)
    # warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
    # for i in range(20):
    #     warmup_scheduler.step()
    #     print(optimizer.state_dict()["param_groups"][0]["lr"])
