import argparse
import json

import mindspore
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

MOE_PARAM_PATTERN = r'.feed_forward.routed_experts.ffn.'
NUM_EXPERTS = 256

def get_moe_file_list(param_map):
    file_set = set()
    for param_name, file_name in param_map.items():
        if MOE_PARAM_PATTERN in param_name:
            file_set.add(file_name)
    return list(file_set)

def get_param_map():
    return json.load(open(f'{args.input_path}/param_name_map.json', 'r'))

def save_param_map(param_map):
    with open(f'{args.output_path}/param_name_map.json', 'w') as f:
        json.dump(param_map, f, indent=4)

def split_moe_params(origin_name, moe_slice):
    ret = dict()
    origin_name_parts = origin_name.split('.')
    for i in range(NUM_EXPERTS):
        new_name = f'{".".join(origin_name_parts[:-3])}.{i}.{".".join(origin_name_parts[-3:])}'
        ret[new_name] = np.transpose(moe_slice[i])
    return ret

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    return parser.parse_args()

def proc_safetensor_file(filename):
    filepath = f'{args.input_path}/{filename}'
    file_params = dict()
    with safe_open(filepath, framework='np') as f:
        for param_name in f.keys():
            if MOE_PARAM_PATTERN in param_name:
                moe_params = split_moe_params(param_name, f.get_slice(param_name))
                file_params.update(moe_params)
            else:
                file_params[param_name] = f.get_tensor(param_name)
    save_file(file_params, f'{args.output_path}/{filename}')

def proc_param_map(param_map):
    for k in tqdm(list(param_map)):
        if MOE_PARAM_PATTERN in k:
            origin_name_parts = k.split('.')
            for i in range(NUM_EXPERTS):
                new_name = f'{".".join(origin_name_parts[:-3])}.{i}.{".".join(origin_name_parts[-3:])}'
                param_map[new_name] = param_map[k]
            del param_map[k]

def main():
    param_map = get_param_map()
    file_map = get_moe_file_list(param_map)
    print(f'Found {len(file_map)} files with MOE parameters.')
    print(f'Processing {len(file_map)} files...')
    process_map(proc_safetensor_file, file_map, max_workers=32)
    print('Processing complete. Now processing parameter map...')
    proc_param_map(param_map)
    save_param_map(param_map)
    print('Done!')

if __name__ == "__main__":
    args = load_args()
    main()
