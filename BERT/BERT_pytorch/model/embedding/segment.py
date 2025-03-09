import torch.nn as nn

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


"""
Segment embeddings are used in transformer-based models (like BERT) to indicate which part of the input a token belongs to. 
This is especially useful in tasks involving pairs of sentences, such as:

Sentence A (Segment 1)
Sentence B (Segment 2)
Padding tokens (Segment 0)"
"""