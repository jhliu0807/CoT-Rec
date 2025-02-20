import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset_name = 'MIND'  # CDs_and_Vinyl Baby_Products Video_Games
mode = 'random'
stage = '700'
p=1

df = pd.read_csv(f'datasets/processed/{dataset_name}.csv', names=['user_id', 'item_id'], usecols=[0, 1])

all_ranks = []
for tar in range(-1, 10):
    filename_base = f'analysis/{dataset_name}_{mode}_{stage}_{tar}_'
    with open(filename_base + f'test_{p}_records.pkl', 'rb') as f:
        records = pickle.load(f)
    with open(filename_base + f'test_{p}_ori_ranks.pkl', 'rb') as f:
        ori_ranks = pickle.load(f)
    with open(filename_base + f'test_{p}_now_ranks.pkl', 'rb') as f:
        now_ranks = pickle.load(f)
    with open(filename_base + f'test_{p}_output_probs.pkl', 'rb') as f:
        output_probs = pickle.load(f)
    ranks = []
    for ori_rank, now_rank, record, prob in zip (ori_ranks, now_ranks, records, output_probs):
        ranks.append(now_rank)
    all_ranks.append(ranks)
all_ranks = np.array(all_ranks)
bias=(abs(all_ranks[1:, :] - all_ranks[1:, :].mean(0))).mean(0).mean()
print(bias)