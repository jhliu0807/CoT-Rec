import torch
import math


class Metrics:
    def __init__(self, topk_list):
        self.topk_list = topk_list
        self.hit_total = {k: 0 for k in self.topk_list}
        self.ndcg_total = {k: 0 for k in self.topk_list}
        self.mrr_total = {k: 0 for k in self.topk_list}
        self.rec_list = []
        self.total_nums = 0

    def get(self):
        hit_total = {k: self.hit_total[k] / self.total_nums for k in self.topk_list}
        ndcg_total = {k: self.ndcg_total[k] / self.total_nums for k in self.topk_list}
        mrr_total = {k: self.mrr_total[k] / self.total_nums for k in self.topk_list}
        return hit_total, ndcg_total, mrr_total, self.rec_list

    def accumulate(self, ranks_list, y, start=0):
        batch_size = len(y)
        for i in range(batch_size):
            ranks, true_item = ranks_list[i], y[i]
            if true_item in ranks:
                rank = ranks.index(true_item) + 1
                self.rec_list.append((start+i, ranks, true_item))
                for k in self.topk_list:
                    if rank <= k:
                        self.hit_total[k] += 1
                        self.ndcg_total[k] += 1 / math.log2(rank + 1)
                        self.mrr_total[k] += 1 / rank
        self.total_nums += batch_size


def get_top_k_recommendations(scores, all_records, k, phase):
    delta = 2 if phase == 'valid' else 1
    for idx, interacted_items in enumerate(all_records):
        scores[idx, interacted_items[:-delta]] = -torch.inf
    top_scores, top_indices = torch.topk(scores, k, dim=1)
    return top_indices
