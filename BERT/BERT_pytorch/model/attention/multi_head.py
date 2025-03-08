import torch.nn as nn
from .single import Attention

class MultiHeadAttention(nn.Module):
    """
        Take in model size and number of heads.
    """

    def __init__(self,h,d_model,dropout=0.1):
        super.__init__()
        assert d_model%h==0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(3)])

        self.output_layer = nn.Linear(d_model,d_model)

        self.attention = Attention()

        self.dropout = dropout

    def forward(self,query,key,value,mask=None):
        batch_size = query.size(0)

        # Apply Linear Transformation
        query_proj = self.linear_layers[0](query)
        key_proj = self.linear_layers[1](key)
        value_proj = self.linear_layers[2](value)

        # Reshape for Multi-Head Attention
        query_proj = query_proj.view(batch_size, -1, self.h, self.d_k)
        key_proj = key_proj.view(batch_size, -1, self.h, self.d_k)
        value_proj = value_proj.view(batch_size, -1, self.h, self.d_k)

        # Transpose for Attention Calculation

        query_proj = query_proj.transpose(1, 2)
        key_proj = key_proj.transpose(1, 2)
        value_proj = value_proj.transpose(1, 2)

        query, key, value = query_proj, key_proj, value_proj  # Assign the final outputs

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

