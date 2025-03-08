# torch.nn.functional allows direct application of functions without needing to create a layer object.
# Why is torch.nn.functional Useful?
# Stateless Operations: It does not store any parameters like torch.nn.Module, making it efficient when we just need to apply a function without defining a layer.
# More Control: It allows us to manually apply operations like activation functions, which is useful for fine-tuning computations.
# Flexibility: We can apply transformations without needing to create layer objects, simplifying custom model architectures.

import torch.nn as nn
import torch.nn.functional as F
import torch

import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self,query,key,value,mask=None, dropout=None):
        scores = torch.matmul(query,key.transpose(-2,-1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores,dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        
        return torch.matmul(p_attn,value),p_attn