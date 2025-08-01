import argparse
import json
from tqdm import tqdm
import os
import shutil

def get_args():
    parser = argparse.ArgumentParser(description="Select layers from weights")
    parser.add_argument('--in_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_layers', type=int, required=True)
    return parser.parse_args()

def load_param_map(weight_path: str) -> dict:
    param_map_path = f"{weight_path}/param_name_map.json"
    with open(param_map_path, "r") as f:
        param_map = json.load(f)
    return param_map

def save_param_map(weight_path: str, param_map: dict) -> None:
    param_map_path = f"{weight_path}/param_name_map.json"
    with open(param_map_path, "w") as f:
        json.dump(param_map, f, indent=2)

def get_layer_num(param_name):
    param_name_parts = param_name.split(".")
    return int(param_name_parts[2])

def select_layers(args) -> dict:
    param_map = load_param_map(args.in_path)
    for k in list(param_map):
        layer = get_layer_num(k)
        if layer > args.n_layers:
            del param_map[k]
    save_param_map(args.out_path, param_map)
    for i in tqdm(param_map):
        in_file = f"{args.in_path}/{i}"
        out_file = f"{args.out_path}/{i}"
        shutil.copy2(in_file, out_file)

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_path, exist_ok=True)
    select_layers(args)
