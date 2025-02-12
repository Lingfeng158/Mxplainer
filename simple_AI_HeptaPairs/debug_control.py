import numpy as np
import subprocess
import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())

# subprocess.run(["python", "running_script.py"])
p = subprocess.Popen(
    ["python", os.getcwd() + "/simple_AI/__main__WMQ.py"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

with open("simple_AI/debug.txt", "r") as file:
    lines = file.readlines()
# print(lines)
line = lines[0]
p.stdin.write(line.encode())
for i in range(1, len(lines)):
    line = lines[i]
    if line[-1] != "\n":
        line += "\n"
    print("command line {}: {}".format(i, line))
    if len(line) == 1:
        continue
    p.stdin.write(line.encode())
    p.stdin.flush()
    # wait_for_res = True
    # while wait_for_res:
    for j in range(20):
        res = p.stdout.readline().decode()
        # print(res)
        time.sleep(0.1)
        if res != ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n":
            # wait_for_res = True
            print(res)
        else:
            break
        # else:
        # wait_for_res = False
p.stdin.write("stop".encode())
p.stdin.flush()
subprocess.Popen.kill(p)

# input_list = ["1", "2", "3"]
# line = input()
# while line != "stop":
#     p.stdin.write((line + "\n").encode())
#     p.stdin.flush()
#     output = p.stdout.readline()
#     print(output.decode())
#     line = input()
# p.stdin.write(("hello world\n").encode())
# # p.stdout.readline()
# p.stdin.write("stop\n".encode())
