from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader as TorchDataLoader
import torch
import torch.nn as nn


class DataLoader:
    source_vocab = None
    target_vocab = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = get_tokenizer(tokenize_en)
        self.tokenize_de = get_tokenizer(tokenize_de)
        self.init_token = init_token
        self.eos_token = eos_token
        print('Dataset initialization started')

    def yield_tokens(self, data_iter, tokenizer):
        for _, text in data_iter:
            yield tokenizer(text)

    def make_dataset(self):
        train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))

        return list(train_data), list(valid_data), list(test_data)

    def build_vocab(self, train_data, min_freq=2):
        self.source_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_data, self.tokenize_de),
            min_freq=min_freq,
            specials=[self.init_token, self.eos_token, '<unk>', '<pad>']
        )
        self.target_vocab = build_vocab_from_iterator(
            self.yield_tokens(train_data, self.tokenize_en),
            min_freq=min_freq,
            specials=[self.init_token, self.eos_token, '<unk>', '<pad>']
        )

        self.source_vocab.set_default_index(self.source_vocab["<unk>"])
        self.target_vocab.set_default_index(self.target_vocab["<unk>"])

    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_tensor = [self.source_vocab[self.init_token]] + [self.source_vocab[token] for token in self.tokenize_de(src_sample)] + [self.source_vocab[self.eos_token]]
            tgt_tensor = [self.target_vocab[self.init_token]] + [self.target_vocab[token] for token in self.tokenize_en(tgt_sample)] + [self.target_vocab[self.eos_token]]
            src_batch.append(torch.tensor(src_tensor, dtype=torch.long))
            tgt_batch.append(torch.tensor(tgt_tensor, dtype=torch.long))
        src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=self.source_vocab['<pad>'], batch_first=True)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.target_vocab['<pad>'], batch_first=True)
        return src_batch, tgt_batch

    def make_iter(self, train_data, valid_data, test_data, batch_size, device):
        train_iterator = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        valid_iterator = TorchDataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)
        test_iterator = TorchDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        print('Dataset initialization done')
        return train_iterator, valid_iterator, test_iterator
