import argparse
import json

import mindspore
from safetensors import safe_open
from safetensors.numpy import save_file
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

MOE_PARAM_PATTERN = r'.feed_forward.routed_experts.ffn.'
ATTN_K_PARAM_PATTERN = r'.attention.lkv2kv_k_nope.'
ATTN_V_PARAM_PATTERN = r'.attention.lkv2kv_v.'
NUM_EXPERTS = 256

def get_file_list(param_map, pattern):
    file_set = set()
    for param_name, file_name in param_map.items():
        if pattern in param_name:
            file_set.add(file_name)
    return list(file_set)

def get_param_map():
    return json.load(open(f'{args.weight_path}/param_name_map.json', 'r'))

def save_param_map(param_map):
    with open(f'{args.weight_path}/param_name_map.json', 'w') as f:
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
    parser.add_argument('weight_path', type=str)
    return parser.parse_args()

def proc_moe_safetensor_file(filename):
    filepath = f'{args.weight_path}/{filename}'
    file_params = dict()
    with safe_open(filepath, framework='np') as f:
        for param_name in f.keys():
            if MOE_PARAM_PATTERN in param_name:
                moe_params = split_moe_params(param_name, f.get_slice(param_name))
                file_params.update(moe_params)
            else:
                file_params[param_name] = f.get_tensor(param_name)
    save_file(file_params, f'{args.weight_path}/{filename}')

def merge_lkv2kv_params(origin_name, k_tensor, param_map):
    v_name = origin_name.replace(ATTN_K_PARAM_PATTERN, ATTN_V_PARAM_PATTERN)
    v_file = param_map[v_name]
    with safe_open(f'{args.weight_path}/{v_file}', framework='np') as f:
        v_tensor = f.get_tensor(v_name)
    return np.concatenate([k_tensor, v_tensor], axis=0)

def proc_attn_k_safetensor_file(filename, param_map):
    filepath = f'{args.weight_path}/{filename}'
    file_params = dict()
    with safe_open(filepath, framework='np') as f:
        for param_name in f.keys():
            if ATTN_K_PARAM_PATTERN in param_name:
                new_name = param_name.replace('.lkv2kv_k_nope.', '.lkv2kv.')
                new_param = merge_lkv2kv_params(param_name, f.get_tensor(param_name), param_map)
                file_params[new_name] = new_param
            else:
                file_params[param_name] = f.get_tensor(param_name)
    save_file(file_params, f'{args.weight_path}/{filename}')

def proc_attn_v_safetensor_file(filename):
    filepath = f'{args.weight_path}/{filename}'
    file_params = dict()
    with safe_open(filepath, framework='np') as f:
        for param_name in f.keys():
            if ATTN_V_PARAM_PATTERN not in param_name:
                file_params[param_name] = f.get_tensor(param_name)
    save_file(file_params, f'{args.weight_path}/{filename}')

def proc_param_map(param_map):
    for k in tqdm(list(param_map)):
        if MOE_PARAM_PATTERN in k:
            origin_name_parts = k.split('.')
            for i in range(NUM_EXPERTS):
                new_name = f'{".".join(origin_name_parts[:-3])}.{i}.{".".join(origin_name_parts[-3:])}'
                param_map[new_name] = param_map[k]
            del param_map[k]
        if ATTN_K_PARAM_PATTERN in k:
            new_name = k.replace('.lkv2kv_k_nope.', '.lkv2kv.')
            param_map[new_name] = param_map[k]
            del param_map[k]
        if ATTN_V_PARAM_PATTERN in k:
            del param_map[k]


def main():
    param_map = get_param_map()

    file_map = get_file_list(param_map, MOE_PARAM_PATTERN)
    print(f'Found {len(file_map)} files with MOE parameters.')
    print(f'Processing {len(file_map)} files...')
    process_map(proc_moe_safetensor_file, file_map, max_workers=32)

    file_map = get_file_list(param_map, ATTN_K_PARAM_PATTERN)
    print(f'Found {len(file_map)} files with attention K parameters.')
    print(f'Processing {len(file_map)} files...')
    for i in tqdm(file_map):
        proc_attn_k_safetensor_file(i, param_map)

    file_map = get_file_list(param_map, ATTN_V_PARAM_PATTERN)
    print(f'Found {len(file_map)} files with attention V parameters.')
    print(f'Processing {len(file_map)} files...')
    process_map(proc_attn_v_safetensor_file, file_map, max_workers=32)

    print('Processing complete. Now processing parameter map...')
    proc_param_map(param_map)
    save_param_map(param_map)
    print('Done!')

if __name__ == "__main__":
    args = load_args()
    main()
