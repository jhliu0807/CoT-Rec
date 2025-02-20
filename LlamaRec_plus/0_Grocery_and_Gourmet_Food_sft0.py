import os
import json
import pickle
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# Define variables directly
args_dataset_name = "Grocery_and_Gourmet_Food"
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


def build_request(user, rec_list, target, phase):
    delta = 2 if phase == 'valid' else 1

    example_history = (
        "Frontier Co-op Ground Chipotle, 1-Pound Bulk\n"
        "SunButter No Sugar Added Sunflower Butter\n"
        "SweetLeaf Stevia Sweet Drops Lemon Drop\n"
        "Frontier Co-op Cinnamon Powder, Ceylon\n"
        "SweetLeaf Sweet Drops Stevia Clear\n"
        "ALTOIDS Arctic Peppermint Mints\n"
        "Organic Cacao Powder, 1lb\n"
        "RX Nut Butter, 6 Flavor Variety Pack\n"
        "Watkins Pure Almond Extract\n"
        "NuNaturals Stevia Syrup\n"
    )
    example_candidates = (
        "A. Shrewd Food Protein Puffs\n"
        "B. Carbquik Biscuit & Baking Mix\n"
        "C. ChocZero's Strawberry Sugar-Free Syrup\n"
        "D. Lakanto Sugar Free Maple Syrup\n"
        "E. 4th & Heart Himalayan Pink Salt Grass-Fed Ghee\n"
        "F. Amazon Brand - Solimo Medium Roast Coffee Pods\n"
        "G. ChocZero's Keto Bark\n"
        "H. Swerve Sweetener, Confectioners\n"
        "I. Victor Allen's Coffee Caramel Macchiato\n"
        "J. Lakanto Golden Monk Fruit Sweetener\n"
    )

    example_output = (
        "{\n"
        "  \"user_history_perception\": {\n"
        "    \"Frontier Co-op Ground Chipotle, 1-Pound Bulk\": \"Smoked dried chili powder with a rich smoky and earthy aroma, suitable for Southwest and Mexican cuisine.\",\n"
        "    \"SunButter No Sugar Added Sunflower Butter\": \"Sugar-free sunflower butter with natural flavor, nutritious and suitable as a healthy snack or spread.\",\n"
        "    \"SweetLeaf Stevia Sweet Drops Lemon Drop\": \"Liquid stevia drops with zero calories, sugar-free, and a hint of lemon, ideal as a healthy alternative for beverages or baking.\",\n"
        "    \"Frontier Co-op Cinnamon Powder, Ceylon\": \"Organic Ceylon cinnamon powder with a fresh and sweet aroma, certified natural, commonly used in baking, beverages, and desserts.\",\n"
        "    \"SweetLeaf Sweet Drops Stevia Clear\": \"Liquid stevia drops with zero calories and sugar-free, suitable for low-carb or sugar-free diets.\",\n"
        "    \"ALTOIDS Arctic Peppermint Mints\": \"Portable peppermint mints with a cooling flavor, useful as a snack or breath freshener.\",\n"
        "    \"Organic Cacao Powder, 1lb\": \"Unsweetened cacao powder with a rich dark chocolate flavor, certified natural, ideal for baking and beverages.\",\n"
        "    \"RX Nut Butter, 6 Flavor Variety Pack\": \"Nut butter in small packages, high protein, low sugar, and available in various flavors, convenient for healthy snacking.\",\n"
        "    \"Watkins Pure Almond Extract\": \"High-quality almond extract with a rich aroma, suitable for baking or beverage flavoring.\",\n"
        "    \"NuNaturals Stevia Syrup\": \"Plant-based zero-calorie syrup, sugar-free, suitable as a healthy substitute for desserts and beverages.\"\n"
        "  },\n"
        "  \"user_preferences\": \"The user prefers sugar-free, natural foods, focusing on healthy sweeteners, seasonings, and snacks. They are possibly pursuing weight loss or a low-carb diet, emphasizing portability and variety.\",\n"
        "  \"candidate_temp_perception\": {\n"
        "    \"Shrewd Food Protein Puffs\": \"High-protein, low-carb, gluten-free healthy snack. [Comment:] As a user, I find this snack very convenient and nutritious, perfectly fitting my dietary habits.\",\n"
        "    \"Carbquik Biscuit & Baking Mix\": \"Low-carb baking mix suitable for making various low-sugar pastries. [Comment:] I think this product is ideal for creating healthy, low-sugar baked goods and perfectly aligns with my needs.\",\n"
        "    \"ChocZero's Strawberry Sugar-Free Syrup\": \"Sugar-free strawberry-flavored syrup. [Comment:] This syrup is an excellent addition to my low-sugar diet and is highly practical.\",\n"
        "    \"Lakanto Sugar Free Maple Syrup\": \"Sugar-free maple syrup, low-carb and natural sweetener. [Comment:] I feel this maple syrup works wonderfully in beverages or baking and aligns well with my healthy eating goals.\",\n"
        "    \"4th & Heart Himalayan Pink Salt Grass-Fed Ghee\": \"Natural lactose-free grass-fed ghee. [Comment:] This ghee makes me feel connected to natural and healthy cooking, a perfect choice for wholesome meals.\",\n"
        "    \"Amazon Brand - Solimo Medium Roast Coffee Pods\": \"Medium roast coffee pods convenient for quick coffee preparation. [Comment:] While convenient, this product does not meet my low-sugar dietary focus, so I might not prioritize it.\",\n"
        "    \"ChocZero's Keto Bark\": \"Sugar-free dark chocolate snack, low-carb with natural ingredients. [Comment:] I love this healthy sugar-free snack; it tastes amazing!\",\n"
        "    \"Swerve Sweetener, Confectioners\": \"Sugar-free sweetener powder suitable for low-carb and sugar-free baking. [Comment:] As a user, I think this is a perfect sugar substitute and highly practical.\",\n"
        "    \"Victor Allen's Coffee Caramel Macchiato\": \"Caramel macchiato coffee pods convenient for consumption. [Comment:] This product might not fit my dietary preferences due to its sugar content.\",\n"
        "    \"Lakanto Golden Monk Fruit Sweetener\": \"Sugar-free monk fruit sweetener, low-carb and zero-calorie. [Comment:] This is one of my favorite healthy sweeteners, ideal for baking or beverages.\"\n"
        "  },\n"
        "  \"candidate_perception\": {\n"
        "    \"Shrewd Food Protein Puffs\": \"Convenient and nutritious snacks\",\n"
        "    \"Carbquik Biscuit & Baking Mix\": \"Low-carb baking mix\",\n"
        "    \"ChocZero's Strawberry Sugar-Free Syrup\": \"Low-sugar alternative sweetener\",\n"
        "    \"Lakanto Sugar Free Maple Syrup\": \"Natural and low-carb sweetener\",\n"
        "    \"4th & Heart Himalayan Pink Salt Grass-Fed Ghee\": \"Natural and wholesome cooking ingredient\",\n"
        "    \"Amazon Brand - Solimo Medium Roast Coffee Pods\": \"Convenient but lacks health focus\",\n"
        "    \"ChocZero's Keto Bark\": \"Healthy sugar-free snack\",\n"
        "    \"Swerve Sweetener, Confectioners\": \"Excellent sugar substitute\",\n"
        "    \"Victor Allen's Coffee Caramel Macchiato\": \"Convenient but contains sugar\",\n"
        "    \"Lakanto Golden Monk Fruit Sweetener\": \"Ideal for low-carb and healthy baking\"\n"
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
        f"This is a sequential recommendation task involving grocery and gourmet food preferences. Given a user's grocery interaction history and a set of candidate items for the next interaction, your task is as follows:\n\n"
        f"1. Provide an objective description of each item in the user's interaction history, focusing on factual features such as ingredients, health benefits, or notable qualities of each item.\n"
        f"2. Based on these descriptions, predict the user's overall preferences and describe their likely personality and tastes in detail in no more than 80 words.\n"
        f"   - The summarized user preferences should be based on the frequency and regularity of user behavior rather than occasional occurrences.\n"
        f"   - Avoid using generic or vague terms; be specific and relevant.\n"
        f"3. Use the predicted preferences to evaluate each candidate item. Each evaluation must include:\n"
        f"   - Objective features of the item (factual description).\n"
        f"   - User-specific comments based on their preferences, preceded by the tag `[Comment:]` to distinguish them from the factual description.\n"
        f"4. Output the result in JSON format with the following fields:\n"
        f"   - `user_history_perception`: Objective descriptions for items in the user's interaction history.\n"
        f"   - `user_preferences`: A summary of the user's preferences.\n"
        f"   - `candidate_temp_perception`: Evaluations for items in the candidate set, including both factual descriptions and user-specific comments (prefixed with `[Comment:]`).\n"
        f"   - `candidate_perception`: Summarized user-relevant aspects from `candidate_temp_perception` comments, highlighting the most significant point of interest or concern for each item.\n"
        f"5. Ensure the JSON format is strictly correct and complete.\n"
        f"   - Every item in the interaction history and candidate set must be included.\n"
        f"   - Do not omit any items or use ellipses (...).\n"
        f"6. Directly output the JSON format without additional explanations or comments.\n"
        f"7. Strictly follow the format and style in the example provided below. Ensure all required fields are present and formatted correctly.\n\n"
        f"### Example\n"
        f"**User Item Interaction History:**\n{example_history}\n"
        f"**Candidate Items:**\n{example_candidates}\n\n"
        f"**Expected Output:**\n{example_output}\n\n"
        f"### Input\n"
        f"**User Item Interaction History:**\n{history}\n"
        f"**Candidate Items:**\n{candidates}\n\n"
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
