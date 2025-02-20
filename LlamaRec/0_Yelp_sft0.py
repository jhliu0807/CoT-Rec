import os
import json
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Define variables directly
args_dataset_name = "Yelp"
args_mode = "random"
args_k = 10
args_p = 1

random.seed(2025)


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

# tokenizer
model_name = "data/Qwen2_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to build SFT prompts
def build_request(user, rec_list, target, phase):
    delta = 2 if phase == 'valid' else 1

    # Example Interaction History and Candidate Pool
    example_history = (
        "K-POT Korean BBQ & Hot Pot\n"
        "Shogun Japanese Steakhouse & Sushi Bar\n"
        "Sweet Lucy's Smokehouse\n"
        "Three Monkeys Cafe\n"
        "Texas Roadhouse\n"
        "Fran's Pub\n"
        "Miller's Ale House\n"
        "Brickwall Tavern and Dining Room\n"
        "MOD Pizza\n"
    )
    example_candidates = (
        "A. Senor Salsa\n"
        "B. Philadelphia Zoo\n"
        "C. Love & Honey Fried Chicken\n"
        "D. Bernie's Restaurant & Bar - Glenside\n"
        "E. Northeast Sandwich Co.\n"
        "F. LEGOLAND Discovery Center\n"
        "G. Vincent's Pizza\n"
        "H. Ooka Restaurant\n"
        "I. Wheel Works\n"
        "J. Crafty Crab\n"
    )

    example_output = (
        "{\n"
        "  \"user_history_perception\": {\n"
        "    \"K-POT Korean BBQ & Hot Pot\": \"Offers table-side grilling and customizable broths for Korean barbecue and hot pot.\",\n"
        "    \"Shogun Japanese Steakhouse & Sushi Bar\": \"Features hibachi-style cooking performances and traditional sushi options.\",\n"
        "    \"Sweet Lucy's Smokehouse\": \"Specializes in slow-smoked meats like ribs and brisket with signature barbecue sauces.\",\n"
        "    \"Three Monkeys Cafe\": \"Combines creative dishes with classic pub fare on a unique menu.\",\n"
        "    \"Texas Roadhouse\": \"Known for hand-cut steaks, fall-off-the-bone ribs, and cinnamon butter rolls.\",\n"
        "    \"Fran's Pub\": \"Serves burgers, sandwiches, and a variety of bar snacks.\",\n"
        "    \"Miller's Ale House\": \"Sports bar offering zingers, steaks, and seafood dishes.\",\n"
        "    \"Brickwall Tavern and Dining Room\": \"Focuses on house-made comfort classics like meatloaf and pot pies.\",\n"
        "    \"MOD Pizza\": \"Customizable pizzas with fresh toppings and quick-service preparation.\"\n"
        "  },\n"
        "  \"user_preferences\": \"Likely a meat lover, junk food enthusiast, and fan of pub-style dining, while also having a notable interest in Asian cuisine.\",\n"
        "  \"candidate_temp_perception\": {\n"
        "    \"Senor Salsa\": \"Serves Mexican cuisine, including tacos and burritos. [Comment:] These tacos are amazing with just the right amount of flavorful spices. I could eat these all day!\",\n"
        "    \"Philadelphia Zoo\": \"Offers family-friendly activities and light dining options. [Comment:] The snacks here are decent, but I wish they had more meat-heavy options for someone like me.\",\n"
        "    \"Love & Honey Fried Chicken\": \"Specializes in Southern-style fried chicken with crispy coatings. [Comment:] The fried chicken here is absolutely delicious – crispy, juicy, and perfectly seasoned. A must-try!\",\n"
        "    \"Bernie's Restaurant & Bar - Glenside\": \"Features American comfort food and a pub-like atmosphere. [Comment:] This place is great for its classic American dishes. The relaxed vibe really hits the spot after a long day.\",\n"
        "    \"Northeast Sandwich Co.\": \"Creates sandwiches with premium meats and creative fillings. [Comment:] The meat-packed sandwiches here are to die for! Full of flavor and incredibly satisfying.\",\n"
        "    \"LEGOLAND Discovery Center\": \"Offers Lego-themed activities and simple dining options. [Comment:] Fun place, but the food doesn’t really appeal to me. It’s too basic, and I’d love something more flavorful.\",\n"
        "    \"Vincent's Pizza\": \"Serves traditional Italian pizzas with crispy crusts and rich toppings. [Comment:] Their meat-lover’s pizza is phenomenal – crispy crust, generous toppings, and full of savory flavors.\",\n"
        "    \"Ooka Restaurant\": \"Offers sushi and hibachi dishes with an emphasis on fresh ingredients. [Comment:] The hibachi steak is a personal favorite – perfectly cooked and so flavorful. A great spot for a treat!\",\n"
        "    \"Wheel Works\": \"A mechanical-themed venue serving snacks and drinks. [Comment:] The snacks are fine, but I’m more interested in their pub-like food offerings. A good place for a casual bite.\",\n"
        "    \"Crafty Crab\": \"Specializes in seafood boils with customizable flavors. [Comment:] The seafood here is fresh, and the bold spices make every bite a delight. Perfect for spice lovers!\"\n"
        "  },\n"
        "  \"candidate_perception\": {\n"
        "    \"Senor Salsa\": \"Flavorful spices in tacos\",\n"
        "    \"Philadelphia Zoo\": \"Lack of meat-heavy options\",\n"
        "    \"Love & Honey Fried Chicken\": \"Crispiness and seasoning of fried chicken\",\n"
        "    \"Bernie's Restaurant & Bar - Glenside\": \"Classic American dishes\",\n"
        "    \"Northeast Sandwich Co.\": \"Meat-packed sandwiches\",\n"
        "    \"LEGOLAND Discovery Center\": \"Limited food variety\",\n"
        "    \"Vincent's Pizza\": \"Meat pizza flavor\",\n"
        "    \"Ooka Restaurant\": \"Hibachi steak flavor\",\n"
        "    \"Wheel Works\": \"Pub-like food offerings\",\n"
        "    \"Crafty Crab\": \"Freshness and bold spices in seafood\"\n"
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
        f"This is a sequential recommendation task involving restaurant preferences. Given a user's restaurant interaction history and a set of candidate restaurants for the next interaction, your task is as follows:\n\n"
        f"1. Provide an objective description of each restaurant in the user's interaction history, focusing on factual features such as cuisine types, unique offerings, or notable services provided by each restaurant.\n"
        f"2. Based on these descriptions, predict the user's overall preferences and describe their likely personality and tastes in detail in no more than 80 words.\n"
        f"   - The summarized user preferences should be based on the frequency and regularity of user behavior rather than occasional occurrences.\n"
        f"   - Avoid using generic or vague terms; be specific and relevant.\n"
        f"3. Use the predicted preferences to evaluate each candidate restaurant. Each evaluation must include:\n"
        f"   - Objective features of the restaurant (factual description).\n"
        f"   - User-specific comments based on their preferences, preceded by the tag `[Comment:]` to distinguish them from the factual description.\n"
        f"4. Output the result in JSON format with the following fields:\n"
        f"   - `user_history_perception`: Objective descriptions for restaurants in the user's interaction history.\n"
        f"   - `user_preferences`: A summary of the user's preferences.\n"
        f"   - `candidate_temp_perception`: Evaluations for restaurants in the candidate set, including both factual descriptions and user-specific comments (prefixed with `[Comment:]`).\n"
        f"   - `candidate_perception`: Summarized user-relevant aspects from `candidate_temp_perception` comments, highlighting the most significant point of interest or concern for each restaurant.\n"
        f"5. Ensure the JSON format is strictly correct and complete.\n"
        f"   - Every item in the interaction history and candidate set must be included.\n"
        f"   - Do not omit any items or use ellipses (...).\n"
        f"6. Directly output the JSON format without additional explanations or comments.\n"
        f"7. Strictly follow the format and style in the example provided below. Ensure all required fields are present and formatted correctly.\n\n"
        f"### Example\n"
        f"**User Restaurant Interaction History:**\n{example_history}\n"
        f"**Candidate Restaurants:**\n{example_candidates}\n\n"
        f"**Expected Output:**\n{example_output}\n\n"
        f"### Input\n"
        f"**User Restaurant Interaction History:**\n{history}\n"
        f"**Candidate Restaurants:**\n{candidates}\n\n"
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
