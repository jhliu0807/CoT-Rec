import os
import json
import pickle
import random
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from SASRec.utils import Metrics
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for dataset and model settings")
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--mode', type=str, default='random')  # before random aug
    parser.add_argument('--stage', type=str, default='4')  # 1 2 3 4
    parser.add_argument('--tar', type=int, default=-1)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--device', type=str)
    parser.add_argument('--p', type=int,default=0)
    args = parser.parse_args()
    args.model_name = 'Qwen/Qwen2.5-7B-Instruct' if args.mode == 'before' else f'output/{args.dataset_name}_{args.mode}_{args.p}/checkpoint-{args.stage}'
    return args

args = parse_args()

random.seed(2025)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

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


# id2name, df, num_users
with open(f'datasets/processed/{args.dataset_name}.json', 'r') as file:
    id2name = json.load(file)
    id2name = {int(key): value for key, value in id2name.items()}
df = pd.read_csv(f'datasets/processed/{args.dataset_name}.csv', names=['user_id', 'item_id'], usecols=[0, 1])
num_users = df['user_id'].max() + 1


# model tokenizer
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)


import pickle
if args.p==1:
    with open(f'gpt_sft_data/{args.dataset_name}_valid.pkl', 'rb') as file:
        valid_p=pickle.load(file)
    with open(f'gpt_sft_data/{args.dataset_name}_test.pkl', 'rb') as file:
        test_p=pickle.load(file)


def build_in_out(user, rec_list, target, phase,p):
    delta = 2 if phase == 'valid' else 1
    candidates = [id2name[i] for i in rec_list]
    label = chr(65 + rec_list.index(target))
    if p == 1:
        data_p=valid_p if phase =='valid' else test_p
        try:
            user_preferences,candidate_perception=data_p[str(user)]['user_preferences'],data_p[str(user)]['candidate_perception']
        except KeyError:
            return None,None
        candidates = [f"{chr(65 + i)}. {s}: {candidate_perception.get(s,'None')}" for i, s in enumerate(candidates)]
        candidates = '\n'.join(candidates)        
        history = df[df['user_id'] == user]['item_id'].values[-(args.k + delta):-delta]
        history = [id2name[i] for i in history]
        history = '\n'.join(history)
        prompt = (
    f"### Instruction\n"
    f"Given user history in chronological order, recommend an item from the candidate pool. "
    f"Each item in the user history and candidate pool has a personalized perception phrase after the colon (:), "
    f"reflecting the user's subjective view of the item. Consider both the user's preferences and these phrases when making a recommendation. "
    f"**Only** output its index letter (one of A-J).\n\n"
    f"### Input\n"
    f"**User preferences:**\n"
    f"{user_preferences}\n\n"
    f"**User history:**\n"
    f"{history}\n"
    f"**Candidate pool:**\n"
    f"{candidates}\n\n"
    f"### Response\n"
)
        
    else:
        candidates = [f"{chr(65 + i)}. {s}" for i, s in enumerate(candidates)]
        candidates = '\n'.join(candidates)
        history = df[df['user_id'] == user]['item_id'].values[-(args.k + delta):-delta]
        history = [id2name[i] for i in history]
        history = '\n'.join(history)
        prompt =f"### Instruction\nGiven user history in chronological order, recommend an item from the candidate pool. **Only** output its index letter (one of A-J).\n\n### Input\n**User history:**\n{ history }\n**Candidate pool:**\n{ candidates }\n\n### Response\n"
    
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), label























# 获得预测
# def get_pred(text):
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#     outputs = model(**model_inputs, use_cache=False)
#     first_token_logits = outputs.logits[0, -1, :]  # 取最后一个输入token的logits
#     first_token_probs = torch.softmax(first_token_logits, dim=-1)
#     top_prob, top_token_id = torch.topk(first_token_probs, args.k)
#     rank_pred = tokenizer.convert_ids_to_tokens(top_token_id)  # e.g. ['B', 'C', 'G', ...]
#     return rank_pred, top_prob
def get_pred(text):
    letter2token_id={'A': 32, 'B': 33, 'C': 34, 'D': 35, 'E': 36, 'F': 37, 'G': 38, 'H': 39, 'I': 40, 'J': 41}
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**model_inputs)
    last_logits = outputs.logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    letter2prob = {letter: probs[tid].item() for letter, tid in letter2token_id.items()}
    sorted_letters = sorted(letter2prob.items(), key=lambda x: x[1], reverse=True)
    rank_pred = [x[0] for x in sorted_letters]
    top_prob = [x[1] for x in sorted_letters]
    return rank_pred, top_prob




test_data_json=[]
for phase in ['test']:
    data = data_valid if phase == 'valid' else data_test
    up, down, metrics = 0, 0, Metrics([args.k])
    output_probs_list, ori_ranks_list, now_ranks_list = [], [], []       
    for user, rec_list, target in tqdm(data):
        ori_rank = rec_list.index(target) + 1   # 记录原本的排名
        random.shuffle(rec_list)
        ori = rec_list.index(target)  # 打乱后的位置
        if args.tar >= 0:
            tar = args.tar
        else:
            tar = ori
        rec_list[ori], rec_list[tar] = rec_list[tar], rec_list[ori]  # 调整位置
        text, label = build_in_out(user, rec_list, target, phase,args.p)  # 构造input output
        test_data_json.append({"text": text, "label": label})
        if text is None and label is None:
            continue
        rank_pred, top_prob = get_pred(text)  # 获得预测
        try:
            now_rank = rank_pred.index(label) + 1  # 优化后的排名
            if now_rank < ori_rank:
                up += 1
            if now_rank > ori_rank:
                down += 1
            metrics.accumulate([rank_pred], [label])
        except:
            now_rank = args.k
        now_ranks_list.append(now_rank)
        # output_probs_list.append(top_prob.tolist())
        output_probs_list.append(top_prob)
        ori_ranks_list.append(ori_rank)

    with open(f'analysis/{args.dataset_name}_{args.mode}_{args.stage}_{args.tar}_{phase}_{args.p}_ori_ranks.pkl', 'wb') as f:
        pickle.dump(ori_ranks_list, f)
    with open(f'analysis/{args.dataset_name}_{args.mode}_{args.stage}_{args.tar}_{phase}_{args.p}_now_ranks.pkl', 'wb') as f:
        pickle.dump(now_ranks_list, f)
    with open(f'analysis/{args.dataset_name}_{args.mode}_{args.stage}_{args.tar}_{phase}_{args.p}_output_probs.pkl', 'wb') as f:
        pickle.dump(output_probs_list, f)
    with open(f'analysis/{args.dataset_name}_{args.mode}_{args.stage}_{args.tar}_{phase}_{args.p}_records.pkl', 'wb') as f:
        pickle.dump(data, f)

    with open(f'results/{args.dataset_name}_{args.p}.txt', 'a') as f:
        f.write(f'[{args.mode}_{args.stage}_{args.tar}_{phase}]\n')
        f.write(f'提升:{up / len(data):.4f}\n')
        f.write(f'下降:{down / len(data):.4f}\n')
        f.write(f'NDCG:{metrics.ndcg_total[args.k] / num_users:.4f}\n')
        f.write(f'MRR:{metrics.mrr_total[args.k] / num_users:.4f}\n')
test_set_dir = "test_data"
os.makedirs(test_set_dir, exist_ok=True)
with open(f'{test_set_dir}/test_set.json', 'w', encoding='utf-8') as f:
    json.dump(test_data_json, f, ensure_ascii=False, indent=4)