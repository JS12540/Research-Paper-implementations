import torch.nn as nn
import torch
import math

# Transformers lack recurrence (unlike RNNs) and process the input in parallel. 
# To maintain word order information, positional encodings are added to token embeddings.


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False # positional encoding is not learned, but fixed.

        position = torch.arange(0, max_len).float().unsqueeze(1)  # Shape: (max_len, 1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices

        pe = pe.unsqueeze(0)  # Shape becomes (1, max_len, d_model) for batch processing
        self.register_buffer('pe', pe)  # Saves tensor without tracking gradients

    def forward(self, x):
        """The method returns the positional encodings for the corresponding sequence length."""
        return self.pe[:, :x.size(1)]