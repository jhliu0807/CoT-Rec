import pickle
import json
import re
from tqdm import tqdm
import random
import pandas as pd
dataset='Yelp'
with open(f'datasets/processed/{dataset}.json', 'r') as file:
    id2name = json.load(file)
    id2name = {int(key): value for key, value in id2name.items()}
df = pd.read_csv(f'datasets/processed/{dataset}.csv', names=['user_id', 'item_id'], usecols=[0, 1])
def build_openai_format(user, rec_list, target, phase, data_p):
    try:
        user_preferences = data_p[str(user)]['user_preferences']
        candidate_perception = data_p[str(user)]['candidate_perception']
    except KeyError:
        return None

    candidates = [id2name[i] for i in rec_list]
    candidates = [f"{chr(65 + i)}. {s}: {candidate_perception.get(s, 'None')}" for i, s in enumerate(candidates)]
    candidates = '\n'.join(candidates)

    label = chr(65 + rec_list.index(target))

    delta = 2 if phase == 'valid' else 1
    k=10
    history = df[df['user_id'] == user]['item_id'].values[-(k + delta):-delta]
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

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": label}
    ]
    return messages

def process_and_save(input_path_valid, input_path_test, output_file_valid, output_file_test):
    args_k=10
    with open(input_path_valid, 'rb') as file:
        valid_p = pickle.load(file)
    with open(input_path_test, 'rb') as file:
        test_p = pickle.load(file)
    
    with open(f'SASRec/checkpoint/{dataset}_rec_list_valid.pkl', 'rb') as f:
        rec_list_valid = pickle.load(f)
    with open(f'SASRec/checkpoint/{dataset}_rec_list_test.pkl', 'rb') as f:
        rec_list_test = pickle.load(f)
    data_valid = []
    for u, rec_list, i in rec_list_valid:
        if i in rec_list[:args_k]:
            data_valid.append((u, rec_list[:args_k], i))
    data_test = []
    for u, rec_list, i in rec_list_test:
        if i in rec_list[:args_k]:
            data_test.append((u, rec_list[:args_k], i))   
    
    

    for phase, data_p, output_file in [('valid', valid_p, output_file_valid), ('test', test_p, output_file_test)]:
        openai_data = []
        data = data_valid if phase == 'valid' else data_test
        for user, rec_list, target in tqdm(data, desc=f"Processing {phase}"):
            random.shuffle(rec_list)
            messages = build_openai_format(user, rec_list, target, phase, data_p)
            if messages is None:
                continue
            openai_data.append({"messages": messages})
        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(openai_data, outfile, ensure_ascii=False, indent=2)


input_path_valid = f"gpt_sft_data/{dataset}_valid.pkl"
input_path_test = f"gpt_sft_data/{dataset}_test.pkl"
output_file_valid = f"LLaMA-Factory/data/{dataset}_random_valid_1_plus.json"
output_file_test = f"LLaMA-Factory/data/{dataset}_random_test_1_plus.json"

process_and_save(input_path_valid, input_path_test, output_file_valid, output_file_test)
