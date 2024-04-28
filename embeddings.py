import torch
import torch.nn as nn
from src.config import *


class TokenEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """computes the token embedding

        Args:
            d_model (int): embedding dim
            vocab_size (int): size of the vocabulary
        """

        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embedding=self.vocab_size, embedding_dim=d_model
        )  # shape: (vocab_size, embedding_dim)

    def forward(self, x: torch.Tensor):
        token_embedding = self.embedding(x)  # embedding created using embedding class
        return token_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout_p: float) -> None:
        """computer positional encosing for the input tokens positions

        Args:
            d_model (int): embedding dim
            seq_len (int): input token sequence length
            dropout_p (float): dropout value

        Returns:
            pos_encoding: encoded position values
        """

        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        position_encodings = torch.zeros(self.seq_len, self.d_model)
        positions = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        even_odd_i = torch.arange(0, self.d_model, 2).float()
        div_freqs_term = torch.pow(10000, even_odd_i / d_model)
        position_encodings[:, 0::2] = torch.sin(positions * div_freqs_term)
        position_encodings[:, 1::2] = torch.cos(
            positions * div_freqs_term
        )  # till here the size will be (seq_len, d_model)
        position_encodings = position_encodings.unsqueeze(
            0
        )  # doing this unsqueeze because this needs to be a batch : (1, seq_len, d_model)
        self.register_buffer(
            "position_encodings", positions
        )  # registering this as a buffer, this is not a parameter, this should be a part of module and not to be updated during backward prop

    def forward(self, x: torch.Tensor):
        x = x + (self.position_encodings[:, : x.shape[1], :]).requires_grad_(False)
        pos_encoding = self.dropout(x)
        return pos_encoding


class InputEnbeddings(nn.Module):
    def __init__(self) -> None:
        """
        An embedding module which will compute toekn embedding with addition to positional embedding

        Returns:

        """
        super().__init__()
        self.d_model = D_MODEL
        self.vocab_size = VOCAB_SIZE
        self.seq_len = MAX_SEQ_LEN
        self.dropout_p = FF_DROPOUT
        self.token_embedding = TokenEmbeddings(self.d_model, self.vocab_size)
        self.positional_encoding = PositionalEncoding(
            self.d_model, self.seq_len, self.dropout_p
        )

        def forward(self, x: torch.Tensor):
            token_embed_x = self.token_embedding(x)
            inp_embedding = self.positional_encoding(token_embed_x)
            return inp_embedding
