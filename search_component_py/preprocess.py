from feature import DataFormatting
import numpy as np
import json
import os
import time

matchid = -1
path_to_botzone_log = "data_src/data.txt"
with open(path_to_botzone_log, encoding="UTF-8") as f:
    line = f.readline()
    while line:
        t = line.split()
        if len(t) == 0:
            line = f.readline()
            continue
        if t[0] == "Match":
            data_recorder = DataFormatting(t[1])
            matchid += 1
            if matchid % 1024 == 0:
                print("Processing match %d %s..." % (matchid, t[1]))
        elif t[0] == "Score":
            data_recorder.request2obs(line)
            data_recorder.save_log("data/", "{}.npy".format(matchid))
        else:
            data_recorder.request2obs(line)
        line = f.readline()

# data_recorder.trial()
