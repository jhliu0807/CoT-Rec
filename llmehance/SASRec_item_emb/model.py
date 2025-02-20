import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRec(nn.Module):
    def __init__(self, pretrained_embeddings, num_items, embedding_dim=64, max_length=50, num_layers=2, dropout=0.2, std=1e-3):
        super(SASRec, self).__init__()
        self.std = std
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=num_items)  # padding
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.attn_layers = nn.ModuleList([TransformerLayer(embedding_dim, dropout) for _ in range(num_layers)])
        self.apply(self._init_weights)
        self.item_embedding.weight.data[:num_items] = pretrained_embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, item_ids):
        batch_size, seq_len = item_ids.shape                                                          # [batch_size, seq_len]
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).repeat(batch_size, 1)  # [batch_size, seq_len]
        item_embeds = self.item_embedding(item_ids)      # [batch_size, seq_len, embedding_dim]
        pos_embeds = self.position_embedding(positions)  # [batch_size, seq_len, embedding_dim]
        x = item_embeds + pos_embeds
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        logits = torch.matmul(x, self.item_embedding.weight[:-1].T)  # [batch_size, seq_len, num_items]
        return logits


class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(TransformerLayer, self).__init__()
        self.attn = UnidirectionalSelfAttention(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(x)
        x = self.layer_norm(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.layer_norm(x + self.dropout(ff_output))
        return x


class UnidirectionalSelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(UnidirectionalSelfAttention, self).__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        seq_len, embedding_dim = x.size(1), x.size(2)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (embedding_dim ** 0.5)
        mask = torch.tril(torch.ones(seq_len, seq_len)).bool().unsqueeze(0).to(attention_scores.device)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        attention_output = torch.matmul(attention_weights, V)    # (batch_size, seq_len, embedding_dim)
        return attention_output
