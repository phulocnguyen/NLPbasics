import torch
from Transformer import *
from data.data import *

src_vocab_size = 11
target_vocab_size = 11
num_layers = 6
seq_length= 12


# let 0 be sos token and 1 be eos token
src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9, 10, 1], 
                    [0, 2, 8, 7, 3, 4, 5, 6, 7, 2, 10, 1]])
target = torch.tensor([[0, 1, 7, 4, 3, 5, 9, 2, 8, 10, 9, 1], 
                       [0, 1, 5, 6, 2, 4, 7, 6, 2, 8, 10, 1]])


print(src.shape,target.shape)



model = Transformer(src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size=src_vocab_size, 
                    dec_voc_size=target_vocab_size, d_model=512, n_head=8, max_len=seq_length, 
                    hidden_units=8, num_layers=num_layers, drop_prob=0.2)

model.summary()



# src_pad_idx = loader.source.vocab.stoi['<pad>']
# trg_pad_idx = loader.target.vocab.stoi['<pad>']
# trg_sos_idx = loader.target.vocab.stoi['<sos>']