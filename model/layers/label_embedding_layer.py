import torch
import torch.nn as nn


class GramEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config['word_encoder_output_dim'], len(config['vocab'].vocab), bias=False)

    def forward(self, hidden_grammeme):
        output = self.linear(hidden_grammeme)
        return output
