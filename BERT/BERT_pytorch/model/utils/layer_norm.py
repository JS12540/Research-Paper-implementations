# Layer Normalization helps stabilize and accelerate training by normalizing the input across features (last dimension),
#  instead of across batches (like Batch Normalization).

import torch.nn as nn
import torch

class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).

    Args:
        features (int): Number of input features (last dimension size).
        eps (float, optional): Small constant for numerical stability. Default is 1e-6.

    Learnable Parameters:
        a_2 (torch.nn.Parameter): Scale parameter (γ) initialized to 1.
        b_2 (torch.nn.Parameter): Bias parameter (β) initialized to 0.

    Purpose:
        Layer Normalization normalizes the input across the last dimension,
        helping stabilize training by ensuring consistent input distributions
        across layers.

    Formula:
        LayerNorm(x) = (x - mean) / (std + eps) * γ + β
    """
    
    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # Learnable scale parameter (γ)
        self.b_2 = nn.Parameter(torch.zeros(features))  # Learnable bias parameter (β)
        self.eps = eps  # Small constant to prevent division by zero

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # Mean across last dimension
        std = x.std(-1, keepdim=True)    # Standard deviation across last dimension
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2