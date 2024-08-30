import torch
import torch.nn.functional as F
import math

# Position Embedding
def Position_Embedding(inputs, position_size):
    batch_size, seq_len, word_size = inputs.shape
    position_j = 1. / torch.pow(10000., torch.arange(0, position_size, 2).float() / position_size)
    position_j = position_j.unsqueeze(0)
    position_i = torch.arange(seq_len).unsqueeze(1).float()
    position_ij = torch.matmul(position_i, position_j)
    position_ij = torch.cat([torch.cos(position_ij), torch.sin(position_ij)], dim=1)
    position_embedding = position_ij.unsqueeze(0) + torch.zeros((batch_size, seq_len, position_size))
    return position_embedding

# Mask
def Mask(inputs, seq_len, mode='mul'):
    if seq_len is None:
        return inputs
    else:
        mask = torch.arange(inputs.shape[1])[None, :] < seq_len[:, None]
        mask = mask.float()
        if mode == 'mul':
            return inputs * mask.unsqueeze(-1)
        if mode == 'add':
            return inputs - (1 - mask.unsqueeze(-1)) * 1e12

# Dense Layer
class Dense(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Dense, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(input_size, output_size) * 0.1 - 0.05)
        if bias:
            self.bias = torch.nn.Parameter(torch.rand(output_size) * 0.1 - 0.05)
        else:
            self.bias = None

    def forward(self, inputs):
        outputs = torch.matmul(inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs

# Attention Mechanism
def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    batch_size, seq_len, _ = Q.shape

    # Linear projections
    Q = Dense(Q.shape[-1], nb_head * size_per_head, bias=False)(Q)
    K = Dense(K.shape[-1], nb_head * size_per_head, bias=False)(K)
    V = Dense(V.shape[-1], nb_head * size_per_head, bias=False)(V)
    
    # Reshape and Transpose for Multi-Head Attention
    Q = Q.view(batch_size, seq_len, nb_head, size_per_head).transpose(1, 2)
    K = K.view(batch_size, seq_len, nb_head, size_per_head).transpose(1, 2)
    V = V.view(batch_size, seq_len, nb_head, size_per_head).transpose(1, 2)

    # Scaled Dot-Product Attention
    A = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(size_per_head)
    A = Mask(A, V_len, mode='add')
    A = F.softmax(A, dim=-1)

    # Apply attention to V
    O = torch.matmul(A, V)
    O = O.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    O = Mask(O, Q_len, 'mul')
    return O
