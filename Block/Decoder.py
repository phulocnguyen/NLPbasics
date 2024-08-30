import torch
import torch.nn as nn
from Block.TransformerEmbedding import *
from Layer.LayerNorm import *
from Layer.MultiheadAttention import *
from Layer.PointwiseFeedForward import *

class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_units, num_heads, dropout_probs):
        super(DecoderBlock, self).__init__()
        self.masked_self_attention = MultiheadAttention(num_heads=num_heads, d_model=d_model)
        self.layer_norm = LayerNorm(d_model=d_model)
        self.drop_out = nn.Dropout(p=dropout_probs)
        self.cross_attention = MultiheadAttention(num_heads=num_heads, d_model=d_model)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=hidden_units, drop_prob=dropout_probs)
    
    def forward(self, encoder, decoder, lookahead_mask, source_mask):
        output1 = self.masked_self_attention(q=decoder, k=decoder, v=decoder, mask=lookahead_mask)
        output1 = self.layer_norm(self.drop_out(output1) + decoder)

        if encoder is not None:
            output2 = self.cross_attention(q=decoder, k=encoder, v=encoder, mask=source_mask)
            output2 = self.layer_norm(self.drop_out(output2) + output1)
        
        output3 = self.ffn(output2)
        output3 = self.layer_norm(self.drop_out(output3) + output2)

        return output3


class Decoder(nn.Module):
    def __init__(self, num_heads, d_model, vocab_size, max_len, hidden_units, num_layers, dropout_probs, device):
        super(Decoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size, max_len=max_len, device=device, 
                                              d_model=d_model, dropout_probs=dropout_probs)
        self.linear = nn.Linear(d_model, vocab_size)
        
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model=d_model, num_heads=num_heads, dropout_probs=dropout_probs, 
                                          hidden=hidden_units)
                                          for i in range(num_layers)])
    
    def forward(self, encoder, decoder, lookahead_mask, source_mask):
        output = self.embedding(decoder)
        
        for layer in self.decoder_layers:
            output = layer(encoder, output, lookahead_mask, source_mask)
        
        output = self.linear(output)
        
        return output