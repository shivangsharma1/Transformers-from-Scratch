import torch
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim, attention_dim):
        super().__init__()
        torch.manual_seed(0)
        self.query_weight = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.key_weight = nn.Linear(embedding_dim, attention_dim, bias = False)
        self.value_weight = nn.Linear(embedding_dim, attention_dim, bias = False)

    def forward(self, embedded):
        query = self.query_weight(embedded)
        key = self.key_weight(embedded)
        value = self.value_weight(embedded)

        scores = query @ torch.transpose(query, 1, 2)
        _ ,context_length, attention_dim = key.shape
        scores = scores / (attention_dim ** 0.5)

        premask = torch.tril(torch.ones(context_length, context_length))
        mask = premask == 0
        scores = scores.masked_fill(mask, float('-inf'))
        scores = nn.functional.softmax(scores, dim=2)

        return scores @ value



if __name__ == '__main__':
    embedded = torch.rand(2, 2, 2)
    embed_dim, attention_dim = 2, 3
    obj = SingleHeadAttention(embed_dim, attention_dim)
    out = obj(embedded)
    print(out)