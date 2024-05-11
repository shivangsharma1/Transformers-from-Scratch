import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from vanilla_nn import VanillaNN

class TransformerBlock(nn.Module):
    def __init__(self, attention_dim, num_head) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.mhsa = MultiHeadAttention(attention_dim, num_head)
        self.first_layer_norm = nn.LayerNorm(attention_dim)
        self.second_layer_norm = nn.LayerNorm(attention_dim)
        self.ff = VanillaNN(attention_dim)

    def forward(self, embedded):
        torch.manual_seed(0)
        first_part = embedded + self.mhsa(self.first_layer_norm(embedded))
        result = first_part + self.ff(self.second_layer_norm(first_part))
        return torch.round(result, decimals=4)
    


if __name__ == '__main__':
    embedded = torch.rand(2, 2, 6)
    attention_dim, num_heads = 6, 3

    obj = TransformerBlock(attention_dim, num_heads)
    out = obj(embedded)
    print(out)

