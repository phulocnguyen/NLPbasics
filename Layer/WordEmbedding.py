import torch
import torch.nn as nn

class WordEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(WordEmbedding, self).__init__(num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=1)