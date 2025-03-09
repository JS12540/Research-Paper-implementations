import torch.nn as nn
from .gelu import GELU

class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self,d_model,d_ff,dropout=0.1):
        """
        Initializes the Position-wise Feed-Forward Network (FFN) used in Transformers.

        Args:
            d_model (int): The input and output dimensionality of the FFN (e.g., 768 for BERT).
            d_ff (int): The hidden layer dimensionality, typically 4x d_model (e.g., 3072 for BERT).
            dropout (float, optional): Dropout probability to prevent overfitting. Default is 0.1.

        Components:
            - Linear layer (d_model → d_ff) to expand the feature space.
            - GELU activation function for non-linearity.
            - Dropout layer for regularization.
            - Linear layer (d_ff → d_model) to project back to original size.

        """
        super(PositionWiseFeedForward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self,x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))