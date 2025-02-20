import os
import json
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Define variables directly
args_dataset_name = "MIND"
args_mode = "random"
args_k = 10
args_p = 1

random.seed(2025)
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# indices_head, indices_tail
if args_mode == 'aug3':
    indices_head = np.load(f'round2/{args_dataset_name}_indices_head.npy')
    indices_tail = np.load(f'round2/{args_dataset_name}_indices_tail.npy')

# data_valid, data_test
with open(f'SASRec/checkpoint/{args_dataset_name}_rec_list_valid.pkl', 'rb') as f:
    rec_list_valid = pickle.load(f)
with open(f'SASRec/checkpoint/{args_dataset_name}_rec_list_test.pkl', 'rb') as f:
    rec_list_test = pickle.load(f)
data_valid = []
for u, rec_list, i in rec_list_valid:
    if i in rec_list[:args_k]:
        data_valid.append((u, rec_list[:args_k], i))
data_test = []
for u, rec_list, i in rec_list_test:
    if i in rec_list[:args_k]:
        data_test.append((u, rec_list[:args_k], i))

# id2name, df
with open(f'datasets/processed/{args_dataset_name}.json', 'r') as file:
    id2name = json.load(file)
    id2name = {int(key): value for key, value in id2name.items()}
df = pd.read_csv(f'datasets/processed/{args_dataset_name}.csv', names=['user_id', 'item_id'], usecols=[0, 1])


# Function to build SFT prompts
def build_request(user, rec_list, target, phase):
    delta = 2 if phase == 'valid' else 1

    # Example Interaction History and Candidate Pool
    example_history = (
        "James Mattis mocks Trump's bone spurs\n"
        "Week 7 winners, losers: Aaron Rodgers now in MVP race\n"
        "Four flight attendants were arrested in Miami's airport\n"
        "California sheriff's deputy shot dead, ride-along injured\n"
        "Mitch McConnell snubbed by Elijah Cummings' pallbearers\n"
    )
    example_candidates = (
        "Snow crab sells for record price in Japan\n"
        "One of FBI's Most Wanted fugitives offers surrender\n"
        "Kendall Jenner Wore the Tiniest Dress to Go Jewelry Shopping\n"
        "Far-reaching snowstorm may take shape over US\n"
        "Jimmy Garoppolo addresses Erin Andrews interview by saying he uses 'baby' 500 times a game\n"
    )

    example_output = (
        "{\n"
        "  \"user_history_perception\": {\n"
        "    \"James Mattis mocks Trump's bone spurs\": \"Political satire mocking Trump's military service claim.\",\n"
        "    \"Week 7 winners, losers: Aaron Rodgers now in MVP race\": \"Sports analysis of Aaron Rodgers' performance and MVP chances.\",\n"
        "    \"Four flight attendants were arrested in Miami's airport\": \"Incident involving the arrest of four flight attendants at Miami airport.\",\n"
        "    \"California sheriff's deputy shot dead, ride-along injured\": \"Tragic event involving the death of a sheriff's deputy and injury of a ride-along officer.\",\n"
        "    \"Mitch McConnell snubbed by Elijah Cummings' pallbearers\": \"Political symbolism behind Mitch McConnell's exclusion from Elijah Cummings' funeral procession.\"\n"
        "  },\n"
        "  \"user_preferences\": \"Interested in political controversies, sports updates, and dramatic incidents.\",\n"
        "  \"candidate_temp_perception\": {\n"
        "    \"Snow crab sells for record price in Japan\": \"Luxury market trend as snow crabs reach record prices in Japan. [Comment:] This kind of news fascinates me, as it reflects society's interest in luxury lifestyles through food pricing trends.\",\n"
        "    \"One of FBI's Most Wanted fugitives offers surrender\": \"Crime news involving the surrender of a wanted fugitive. [Comment:] I'm intrigued by these stories, as high-profile criminal cases often come with thought-provoking backgrounds.\",\n"
        "    \"Kendall Jenner Wore the Tiniest Dress to Go Jewelry Shopping\": \"Celebrity fashion feature on Kendall Jenner's unique shopping outfit. [Comment:] While I'm not particularly into fashion, celebrity news like this always manages to grab attention.\",\n"
        "    \"Far-reaching snowstorm may take shape over US\": \"Weather forecast predicting a major snowstorm across the US. [Comment:] This kind of news gets me thinking about the impacts of weather and how well-prepared people are for it.\",\n"
        "    \"Jimmy Garoppolo addresses Erin Andrews interview by saying he uses 'baby' 500 times a game\": \"Sports humor, with Jimmy Garoppolo discussing his frequent use of the term 'baby' in interviews. [Comment:] I find lighthearted sports news like this refreshing, especially as a break from intense game coverage.\"\n"
        "  },\n"
        "  \"candidate_perception\": {\n"
        "    \"Snow crab sells for record price in Japan\": \"Luxury food pricing trends\",\n"
        "    \"One of FBI's Most Wanted fugitives offers surrender\": \"High-profile criminal cases\",\n"
        "    \"Kendall Jenner Wore the Tiniest Dress to Go Jewelry Shopping\": \"Celebrity culture and fashion\",\n"
        "    \"Far-reaching snowstorm may take shape over US\": \"Impactful weather events\",\n"
        "    \"Jimmy Garoppolo addresses Erin Andrews interview by saying he uses 'baby' 500 times a game\": \"Lighthearted sports commentary\"\n"
        "  }\n"
        "}"
    )



    # Current Interaction History and Candidate Pool
    candidates = [id2name[i] for i in rec_list]
    candidates = '\n'.join(candidates)

    history = df[df['user_id'] == user]['item_id'].values[-(args_k + delta):-delta]
    history_ = []
    for item_id in history:
        item_name = id2name[item_id]
        history_.append(f"{item_name}")
    history = '\n'.join(history_)

    # Construct prompt
    prompt = (
        f"### Instruction\n"
        f"This is a sequential recommendation task involving news article preferences. Given a user's news interaction history and a set of candidate news articles for the next interaction, your task is as follows:\n\n"
        f"1. Provide an objective description of each news article in the user's interaction history, focusing on factual features such as topics, unique insights, or notable events covered by each article.\n"
        f"2. Based on these descriptions, predict the user's overall preferences and describe their likely interests and focus areas in detail in no more than 80 words.\n"
        f"   - The summarized user preferences should be based on patterns and regularities in the user's reading behavior rather than occasional occurrences.\n"
        f"   - Avoid using generic or vague terms; be specific and relevant.\n"
        f"3. Use the predicted preferences to evaluate each candidate news article. Each evaluation must include:\n"
        f"   - Objective features of the article (factual description).\n"
        f"   - User-specific comments based on their preferences, preceded by the tag `[Comment:]` to distinguish them from the factual description.\n"
        f"4. Output the result in JSON format with the following fields:\n"
        f"   - `user_history_perception`: Objective descriptions for articles in the user's interaction history.\n"
        f"   - `user_preferences`: A summary of the user's preferences.\n"
        f"   - `candidate_temp_perception`: Evaluations for articles in the candidate set, including both factual descriptions and user-specific comments (prefixed with `[Comment:]`).\n"
        f"   - `candidate_perception`: Summarized user-relevant aspects from `candidate_temp_perception` comments, highlighting the most significant point of interest or concern for each article.\n"
        f"5. Ensure the JSON format is strictly correct and complete.\n"
        f"   - Every item in the interaction history and candidate set must be included.\n"
        f"   - Do not omit any items or use ellipses (...).\n"
        f"6. Directly output the JSON format without additional explanations or comments.\n"
        f"7. Strictly follow the format and style in the example provided below. Ensure all required fields are present and formatted correctly.\n\n"
        f"### Example\n"
        f"**User News Interaction History:**\n{example_history}\n"
        f"**Candidate News Articles:**\n{example_candidates}\n\n"
        f"**Expected Output:**\n{example_output}\n\n"
        f"### Input\n"
        f"**User News Interaction History:**\n{history}\n"
        f"**Candidate News Articles:**\n{candidates}\n\n"
        f"### Output\n"
    )


    return prompt


# Save data in chunks of 45000 entries
max_entries_per_file = 45000
for phase in ['valid', 'test']:
    data = data_valid if phase == 'valid' else data_test
    file_index = 1
    entries = []

    for idx, (user, rec_list, target) in tqdm(enumerate(data), desc=f"Processing {phase} data"):
        random.shuffle(rec_list)
        if args_mode == 'random':
            data_entry = {
                "custom_id": str(user),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": build_request(user, rec_list, target, phase)}
                    ]
                    # "response_format":{"type": "json_object"}
                }
            }
            entries.append(data_entry)

            if len(entries) == max_entries_per_file:
                output_file = f'gpt_sft_data/{args_dataset_name}_{args_mode}_{phase}_part{file_index}.jsonl'
                with open(output_file, 'w', encoding='utf-8') as file:
                    for entry in entries:
                        file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                entries = []
                file_index += 1

    # Save remaining entries
    if entries:
        output_file = f'gpt_sft_data/{args_dataset_name}_{args_mode}_{phase}_part{file_index}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as file:
            for entry in entries:
                file.write(json.dumps(entry, ensure_ascii=False) + '\n')
