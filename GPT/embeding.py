import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import argparse

# 配置 OpenAI API
os.environ['OPENAI_API_KEY'] = ''
client = OpenAI()

# 解析命令行参数
parser = argparse.ArgumentParser(description="Process embeddings for a dataset.")
parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to process.")
parser.add_argument("--stage", type=str, required=True)
parser.add_argument("--dim", type=int, required=True)
args = parser.parse_args()

class Args:
    model, dim, batch_size = 'text-embedding-3-large', args.dim, 512
    dataset_name = args.dataset_name
    stage=args.stage
    perception = 0  # TODO: perception 的含义是是否感知，还未实现感知 user_perception Video_Games
    if stage == 'user':
        input_file, output_file = f'input/user/{dataset_name}_user.json', f'output_{args.dim}/user/{dataset_name}_user.npy'
    elif stage == 'feature':    
        input_file, output_file = f'input/feature/{dataset_name}_feature.json', f'output_{args.dim}/feature/{dataset_name}_feature.npy'
    elif stage == 'caption':    
        input_file, output_file = f'input/caption/{dataset_name}_caption.json', f'output_{args.dim}/caption/{dataset_name}_caption.npy'

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def get_embeddings_batch(texts, dim=64, model="text-embedding-3-small"):
    texts = [text.replace("\n", " ") for text in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [normalize_l2(response.data[i].embedding[:dim]) for i in range(len(response.data))]

# 读取输入文件
with open(Args.input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(list(data.items()), columns=['id', 'text'])
df['id'] = df['id'].astype(int)

# 检查 text 是否为空字符串，并替换为 None
df['text'] = df['text'].apply(lambda x: 'None' if x == "" else x)

# 按照 id 排序
df = df.sort_values(by='id')

embeddings = []
for i in tqdm(range(0, len(df), Args.batch_size)):
    batch_texts = df['text'][i:i + Args.batch_size].tolist()
    try:
        batch_embeddings = get_embeddings_batch(batch_texts, dim=Args.dim, model=Args.model)
    except Exception as e:
        print(f"异常:{e}")
        print(batch_texts)
        raise e("异常信息")
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings)
np.save(Args.output_file, embeddings)
