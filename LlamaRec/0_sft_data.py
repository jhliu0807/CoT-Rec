import os
import json
import pickle
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for dataset and model settings")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--p', type=int, default=0)
    args = parser.parse_args()
    return args

args = parse_args()

random.seed(2025)
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# indices_head, indices_tail
if args.mode == 'aug3':
    indices_head = np.load(f'round2/{args.dataset_name}_indices_head.npy')
    indices_tail = np.load(f'round2/{args.dataset_name}_indices_tail.npy')


# data_valid, data_test
with open(f'SASRec/checkpoint/{args.dataset_name}_rec_list_valid.pkl', 'rb') as f:
    rec_list_valid = pickle.load(f)
with open(f'SASRec/checkpoint/{args.dataset_name}_rec_list_test.pkl', 'rb') as f:
    rec_list_test = pickle.load(f)
data_valid = []
for u, rec_list, i in rec_list_valid:
    if i in rec_list[:args.k]:
        data_valid.append((u, rec_list[:args.k], i))
data_test = []
for u, rec_list, i in rec_list_test:
    if i in rec_list[:args.k]:
        data_test.append((u, rec_list[:args.k], i))


# id2name, df
with open(f'datasets/processed/{args.dataset_name}.json', 'r') as file:
    id2name = json.load(file)
    id2name = {int(key): value for key, value in id2name.items()}
df = pd.read_csv(f'datasets/processed/{args.dataset_name}.csv', names=['user_id', 'item_id'], usecols=[0, 1])





def build_sft(user, rec_list, target, phase, p):
    delta = 2 if phase == 'valid' else 1
    candidates = [id2name[i] for i in rec_list]
    candidates = [f"{chr(65 + i)}. {s}" for i, s in enumerate(candidates)]
    candidates = '\n'.join(candidates)
    label = chr(65 + rec_list.index(target))
    history = df[df['user_id'] == user]['item_id'].values[-(args.k + delta):-delta]
    history = [id2name[i] for i in history]
    history = '\n'.join(history)
    prompt =f"### Instruction\nGiven user history in chronological order, recommend an item from the candidate pool. **Only** output its index letter (one of A-J).\n\n### Input\n**User history:**\n{ history }\n**Candidate pool:**\n{ candidates }\n\n### Response\n"
    

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": label}
    ]

    return messages










for phase in ['valid', 'test']:
    openai_data = []
    data = data_valid if phase == 'valid' else data_test
    for idx, (user, rec_list, target) in tqdm(enumerate(data)):
        random.shuffle(rec_list)
        if args.mode == 'random':
            messages=build_sft(user, rec_list, target, phase,args.p)
            openai_data.append({"messages": messages})
    with open(f'LLaMA-Factory/data/{args.dataset_name}_{args.mode}_{phase}_{args.p}.json', 'w', encoding='utf-8') as outfile:
        json.dump(openai_data, outfile, ensure_ascii=False, indent=2)

