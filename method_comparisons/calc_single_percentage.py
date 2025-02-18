import os
import json

if __name__ == "__main__":
    d = None
    with open("data/count.json", "r") as f:
        d = json.load(f)
    full_data = sum(d)
    with open("data/count2.json", "r") as f:
        d = json.load(f)
    filtered_data = sum(d)
    print(filtered_data / full_data, filtered_data, full_data)
