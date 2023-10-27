import torch
import torch.nn as nn


class ConverterWordDecoderRNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(len(config['vocab'].vocab), len(config['vocab'].vocab))
        self.rnn_decoder = nn.GRUCell(len(config['vocab'].vocab),
                                      config['word_encoder_output_dim'])
        self.decoder_dropout = torch.nn.Dropout(p=config['decoder_dropout'])
        self.embeddings_dropout = torch.nn.Dropout(p=config['decoder_embeddings_dropout'])

    def forward(self, inp, hidden):
        # inp = [batch]
        # hidden = [batch, features]

        embedded = self.embeddings_dropout(self.embedding(inp))
        # embedded = [batch, emb dim]

        hidden = self.rnn_decoder(embedded, hidden)
        # hidden = [batch, hid dim]

        return self.decoder_dropout(hidden)
