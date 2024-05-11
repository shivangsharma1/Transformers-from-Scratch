import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, attention_dim, num_heads) -> None:
        super().__init__()
        torch.manual_seed(0)

        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(
                self.SingleHeadAttention(attention_dim, attention_dim // num_heads)
            )

    def forward(self, embedded):
        outputs = []
        for head in self.heads:
            outputs.append(head(embedded))
        cat = torch.cat(outputs, dim=2)

        return torch.round(cat, decimals=4)

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim, attention_dim):
            super().__init__()
            torch.manual_seed(0)
            self.query_weight = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.key_weight = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_weight = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded):
            query = self.query_weight(embedded)
            key = self.key_weight(embedded)
            value = self.value_weight(embedded)

            scores = query @ torch.transpose(query, 1, 2)
            _, context_length, attention_dim = key.shape
            scores = scores / (attention_dim**0.5)

            premask = torch.tril(torch.ones(context_length, context_length))
            mask = premask == 0
            scores = scores.masked_fill(mask, float("-inf"))
            scores = nn.functional.softmax(scores, dim=2)

            return scores @ value


if __name__ == '__main__':
    embedded = torch.rand(2, 2, 6)
    attention_dim, num_heads = 6, 3

    obj = MultiHeadAttention(attention_dim, num_heads)
    out = obj(embedded)
    print(out)