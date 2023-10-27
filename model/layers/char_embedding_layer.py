import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from definitions import PAD_INDEX


class CharEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.char_embedding = nn.Embedding(len(config['vocab_chars'].vocab), config['char_embedding_dim'],
                                           padding_idx=PAD_INDEX)
        self.char_rnn = nn.GRU(config['char_embedding_dim'],
                               config['char_rnn_hidden_dim'],
                               config['char_rnn_layers_num'],
                               batch_first=True,
                               bidirectional=config['char_rnn_bidirectional'])

    def forward(self, sentences):
        # sentences = [batch, words, max_word_len]
        char_embeddings = self.char_embedding(sentences)
        word_lengths = torch.count_nonzero(sentences, dim=2).cpu()

        _, char_rnn_words = self.char_rnn(pack_padded_sequence(
            char_embeddings.view(-1, *char_embeddings.shape[-2:]), word_lengths.view(-1), batch_first=True,
            enforce_sorted=False))
        char_rnn_words = torch.cat((char_rnn_words[-2, :, :], char_rnn_words[-1, :, :]), dim=1)
        # return dim: (batch, words, 2 * char_rnn_hidden_dim)
        return char_rnn_words.view(*sentences.shape[:2], -1)

