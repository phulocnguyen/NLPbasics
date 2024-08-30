import torch
import torch.nn as nn
from Layer.PositionalEmbedding import *
from Layer.WordEmbedding import *

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, device, max_len, dropout_probs):
        super(TransformerEmbedding, self).__init__()
        self.word_embedding = WordEmbedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_embedding = PositionalEmbedding(d_model=d_model, max_len=max_len, device=device)
        self.drop_out = nn.Dropout(p=dropout_probs)
    
    def forward(self, input):
        word_emb = self.word_embedding(input)
        pos_emb = self.pos_embedding(input)
        output = self.drop_out(word_emb + pos_emb)
        return output