import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

class ItemSequenceDataset(Dataset):
    def __init__(self, filepath, max_length):
        df = pd.read_csv(filepath, names=['user_id', 'item_id'])
        self.num_users, self.num_items = df['user_id'].max() + 1, df['item_id'].max() + 1
        
        self.all_records = [[] for _ in range(self.num_users)]
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            user_id, item_id = row['user_id'], row['item_id']
            self.all_records[user_id].append(item_id)
        
        print('# Users:', self.num_users)
        print('# Items:', self.num_items)
        print('# Interactions:', len(df))

        X_train, y_train = [], []
        X_valid, y_valid = [], []
        X_test, y_test = [], []
        for user_id_origin, seq in tqdm(enumerate(self.all_records), total=self.num_users):
            user_id=user_id_origin+self.num_items+1
            train_seq = seq[:-2]
            if len(train_seq) < max_length:
                X_train.append( (max_length - len(train_seq) + 1) * [self.num_items]+[user_id] + train_seq[:-1])
                y_train.append((max_length - len(train_seq) + 1) * [self.num_items] +[self.num_items]+train_seq[1:])
            else:
                for i in range(len(train_seq) - max_length):
                    X_train.append([user_id] + train_seq[i:i+max_length])
                    y_train.append([self.num_items]+train_seq[i+1:i+max_length+1])

            valid_seq = seq[:-1]
            if len(valid_seq) - 1 < max_length:
                X_valid.append((max_length - len(valid_seq) + 1) * [self.num_items]+[user_id] + valid_seq[:-1])
            else:
                X_valid.append([user_id] + valid_seq[-(max_length+1):-1])
            y_valid.append(valid_seq[-1])

            test_seq = seq
            if len(test_seq) - 1 < max_length:
                X_test.append((max_length - len(test_seq) + 1) * [self.num_items] +[user_id] + test_seq[:-1])
            else:
                X_test.append([user_id] + test_seq[-(max_length+1):-1])
            y_test.append(test_seq[-1])

        self.X_train, self.y_train = torch.tensor(X_train), torch.tensor(y_train)
        self.X_valid, self.y_valid = torch.tensor(X_valid), torch.tensor(y_valid)
        self.X_test, self.y_test = torch.tensor(X_test), torch.tensor(y_test)

        print('Data loading completed.')

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
