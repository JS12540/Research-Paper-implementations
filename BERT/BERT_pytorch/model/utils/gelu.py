# For small values, GELU behaves like 0.5𝑥 (like dropout). For large values, GELU behaves like x (like ReLU).

import torch.nn as nn
import torch
import math

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

