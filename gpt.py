import torch
import torch.nn as nn
from transformers_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size, context_len, model_dim, num_blocks, num_heads):
        super().__init__()
        torch.manual_seed(0)
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(context_len, model_dim)
        self.blocks = nn.Sequential()
        for _ in range(num_blocks):
            self.blocks.append(TransformerBlock(model_dim, num_heads))

        self.final_ln = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        torch.manual_seed(0)

        token_embeds = self.token_embedding(context)  # shape: B, t, D
        B, T, D = token_embeds.shape
        positional_embeds = self.pos_embedding(torch.arange(T))
        total_embeddings = token_embeds + positional_embeds

        un_normalized = self.vocab_projection(self.final_ln(self.blocks(total_embeddings)))
        probs = nn.functional.softmax(un_normalized, dim=-1)
        return torch.round(probs, decimals=4)
    

if __name__ == '__main__':
    vocab_size = 5
    context_length = 5
    model_dim = 16
    num_blocks = 4
    num_heads = 4
    # context = [['With', 'great', 'power', 'comes', 'great']]
    context = torch.tensor([[0 , 1 , 2 , 3 , 1 ]])

    obj = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)
    print(obj(context))




