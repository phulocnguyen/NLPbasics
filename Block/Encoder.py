import torch
import torch.nn as nn
from Layer.MultiheadAttention import *
from Layer.LayerNorm import *
from Layer.PointwiseFeedForward import *
from Block.TransformerEmbedding import *

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_probs, hidden):
        super(EncoderBlock, self).__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.attention = MultiheadAttention(num_heads=num_heads, d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=hidden, drop_prob=dropout_probs)
        self.layer_norm = LayerNorm(d_model=d_model, eps=1e-12)
        self.drop_out = nn.Dropout(p=dropout_probs)
    
    def forward(self, q, k, v, mask):
        output1 = self.attention(q, k, v, mask=mask)
        output1 = self.layer_norm(self.drop_out(output1) + q)
        output2 = self.ffn(output1)
        output2 = self.layer_norm(self.drop_out(output2) + output1)
        return output2


class Encoder(nn.Module):
    def __init__(self, num_heads, vocab_size, max_len, d_model, hidden_units, num_layers, dropout_probs, device):
        super(Encoder).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, max_len=max_len, device=device, 
                                              d_model=d_model, dropout_probs=dropout_probs)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model=d_model, num_heads=num_heads, dropout_probs=dropout_probs, 
                                          hidden=hidden_units)
                                          for i in range(num_layers)])
    
    def forward(self, input, mask):
        output = self.embedding(input)
        
        for layer in self.encoder_layers:
            output = layer(output, mask)
        
        return output
        
        



