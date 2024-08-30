import torch
import torch.nn as nn
from Layer.ScaleDotProduct import *

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super(MultiheadAttention, self).__init__()
        self.attention = ScaleDotProduct()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = self.d_model // self.num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # (batch_size, seq_len, d_model) -> # (batch_size, seq_len, num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        # (batch_size, seq_len, num_heads, depth) -> # (batch_size, num_heads, seq_len, depth)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, q, k, v, mask):
        batch_size = q.size(0)

        # Tính toán q, k, v bằng cách áp dụng các lớp Linear tương ứng.
        q = self.W_q(q)  # (batch_size, seq_len, d_model)
        k = self.W_k(k)  # (batch_size, seq_len, d_model)
        v = self.W_v(v)  # (batch_size, seq_len, d_model)

        # Chia q, k, v thành các đầu attention khác nhau.
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        attention_output = self.attention(q, k, v, mask)

        # Kết hợp các heads lại với nhau thành một tensor duy nhất 
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()

        attention_output = attention_output.view(batch_size, -1, self.d_model)

        output = self.W_o(attention_output)

        return output
    
    # Sample input dimensions
batch_size = 4  # Number of samples in the batch
seq_len = 5     # Sequence length (number of tokens in each sample)
d_model = 512     # Dimensionality of the model (input features)
d_k = 64
d_v = 64     
num_heads = 8   # Number of attention heads

# Create random input tensors for query (q), key (k), and value (v)
q = torch.rand(batch_size, seq_len, d_model)  # (batch_size, seq_len_q, d_model)
k = torch.rand(batch_size, seq_len, d_model)  # (batch_size, seq_len_k, d_model)
v = torch.rand(batch_size, seq_len, d_model)  # (batch_size, seq_len_v, d_model)

# Instantiate the MultiHeadAttention module
multi_head_attn = MultiheadAttention(d_model=d_model, num_heads=num_heads, d_k=d_k, d_v=d_v)

# Forward pass through the multi-head attention
output = multi_head_attn(q, k, v)

# Print the outputs
print("Output shape:", output.shape)  # Should be (batch_size, seq_len_q, d_model)
print("Output:", output)


