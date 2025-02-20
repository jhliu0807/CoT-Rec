import argparse
import pickle
import torch
from data import ItemSequenceDataset
from torch.utils.data import DataLoader
from model import SASRec
from torch import optim
from torch import nn
from utils import Metrics, get_top_k_recommendations
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Argument parser for dataset and model settings")
    parser.add_argument('--dataset_name', type=str, default='CDs_and_Vinyl', help="Name of the dataset.")
    parser.add_argument('--embedding_dim', type=int, default=128, help="Dimensionality of the embeddings.")
    parser.add_argument('--max_length', type=int, default=32, help="Maximum length of input sequences.")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the model.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--num_patience', type=int, default=5, help="Early stopping patience.")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate.")
    parser.add_argument('--topk_list', type=int, nargs='+', default=[10, 20, 50, 100], help="List of top-k values for evaluation.")
    parser.add_argument('--verbose', type=int, default=100, help="Interval for verbose output.")
    parser.add_argument('--device', type=int, default=0, help="Device to use for computation (cpu or cuda).")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for reproducibility.")
    parser.add_argument('--iteminit', type=str, default='caption')
    parser.add_argument('--userinit', type=str, default='random')
    args = parser.parse_args()
    args.filepath = f'llmehance/datasets/processed/{args.dataset_name}.csv'
    args.device = torch.device(f'cuda:{args.device}') if args.device >= 0 else torch.device('cpu')
    return args


def train(dataloader, model, loss_func, optimizer, epoch, args):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(args.device), y.to(args.device)
        logits = model(X)                         # [batch_size, seq_len, num_items]
        logits = logits.view(-1, logits.size(2))  # [batch_size * seq_len, num_items)
        y = y.view(-1)                            # [batch_size * seq_len)
        loss = loss_func(logits, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % args.verbose == 0:
            print(f"loss: {train_loss/(batch+1):>7f}  [{batch+1:>5d}/{num_batches:>5d}] epoch: {epoch}")


def test(dataset, model, args, phase):
    X_all, y_all = (dataset.X_valid, dataset.y_valid) if phase == 'valid' else (dataset.X_test, dataset.y_test)
    model.eval()
    metrics = Metrics(args.topk_list)
    with torch.no_grad():
        start = 0
        while True:
            end = start + args.batch_size
            if end > len(y_all):
                end = len(y_all)
            X = X_all[start:end]
            y = y_all[start:end]
            X, y = X.to(args.device), y.to(args.device)
            scores = model(X)[:, -1, :].squeeze(1) # [batch_size, seq_len, num_items] -> [batch_size, 1, num_items] -> [batch_size, num_items]
            ranks_list = get_top_k_recommendations(scores, dataset.all_records[start:end], max(args.topk_list), phase)
            metrics.accumulate(ranks_list.tolist(), y.tolist(), start)
            start += args.batch_size
            if end == len(y_all):
                break
    hit, ndcg, mrr, rec_list = metrics.get()
    print(f'[{phase}]')
    print("Hit:", hit)
    print("NDCG:", ndcg)
    print("MRR:", mrr)
    return hit, ndcg, mrr, rec_list


if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset = ItemSequenceDataset(args.filepath, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    import numpy as np
    if args.iteminit=='feature':
        pretrained_embeddings = torch.from_numpy(np.load(f'pretrain_{args.embedding_dim}/{args.dataset_name}_feature.npy'))
        print(f'itemembedding已加载：pretrain_{args.embedding_dim}/{args.dataset_name}_feature.npy')
    else:
        pretrained_embeddings = torch.from_numpy(np.load(f'pretrain_{args.embedding_dim}/{args.dataset_name}_caption.npy'))
        print(f'itemembedding已加载：pretrain_{args.embedding_dim}/{args.dataset_name}_caption.npy')
    
    
    if args.userinit=='random':
        user_embeddings = None
    else:
        user_embeddings_path = f'pretrain_{args.embedding_dim}/{args.dataset_name}_user.npy'
        user_embeddings = torch.from_numpy(np.load(user_embeddings_path))
    model = SASRec(pretrained_embeddings,user_embeddings, dataset.num_items,dataset.num_users, args.embedding_dim, args.max_length, args.num_layers, dropout=args.dropout).to(args.device)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss(ignore_index=dataset.num_items)

    patience = args.num_patience
    best_ndcg_valid = 0.0
    best_valid, best_test, best_epoch = None, None, None
    test(dataset, model, args, phase='valid')
    test(dataset, model, args, phase='test')
    checkpoint_path=f'llmehance/checkpoint/checkpoint_lr{args.lr}_dim{args.embedding_dim}_numlayers{args.num_layers}_batchsize{args.batch_size}_dropout{args.dropout}'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        print(f"文件夹 '{checkpoint_path}' 已创建。")
    for epoch in range(1, args.num_epochs + 1):
        train(dataloader, model, loss_func, optimizer, epoch, args)
        hit_valid, ndcg_valid, mrr_valid, rec_list_valid = test(dataset, model, args, phase='valid')
        hit_test, ndcg_test, mrr_test, rec_list_test = test(dataset, model, args, phase='test')
        if ndcg_valid[max(args.topk_list)] >= best_ndcg_valid:
            patience = args.num_patience
            best_ndcg_valid = ndcg_valid[max(args.topk_list)]
            best_valid, best_test = (hit_valid, ndcg_valid, mrr_valid), (hit_test, ndcg_test, mrr_test)
            best_epoch = epoch
            torch.save(model, f"{checkpoint_path}/{args.dataset_name}.pth")
            with open(f'{checkpoint_path}/{args.dataset_name}_rec_list_valid.pkl', 'wb') as f:
                pickle.dump(rec_list_valid, f)
            with open(f'{checkpoint_path}/{args.dataset_name}_rec_list_test.pkl', 'wb') as f:
                pickle.dump(rec_list_test, f)
        else:
            patience -= 1
            if patience == 0:
                break
    
    with open(f"{checkpoint_path}/{args.dataset_name}.log", 'a') as f:
        f.write(args.iteminit+'+'+args.userinit)
        f.write('\n')
        f.write(f'best epoch: {best_epoch}\nbest valid: {best_valid}\nbest test: {best_test}\n')
