import torch.nn as nn
from model.layers.char_embedding_layer import CharEmbeddingLayer
import torch


class ConverterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.with_pretrained_word_embeddings = config['with_pretrained_word_embeddings']
        self.char_embedding_layer = CharEmbeddingLayer(config)
        self.embeddings_dropout_layer = torch.nn.Dropout(p=config['word_embeddings_dropout'])
        self.encoder_dropout_layer = torch.nn.Dropout(p=config['word_encoder_dropout'])
        if config['experiment_name'] == 'model_multitask':
            config['word_encoder_output_dim'] = 1600
        elif len(config['vocab'].vocab) == len(config['datasets_names']):
            config['word_encoder_output_dim'] = sum(len(config['vocab'].vocab[i])
                                                    for i in range(len(config['datasets_names'])))
        else:
            config['word_encoder_output_dim'] = len(config['vocab'].vocab) * 2
        self.word_encoder = nn.GRU(config['word_encoder_input_dim'],
                                   config['word_encoder_output_dim'],
                                   batch_first=True,
                                   bidirectional=config['word_encoder_bidirectional'],
                                   num_layers=config['word_encoder_num_layers'])
        config['word_encoder_output_dim'] *= int(config['word_encoder_bidirectional']) + 1

    def forward(self, sentences, word_embeddings):
        # sentences = [batch, words, max_word_len]
        # word_embeddings = [batch, words, embedding_dim]
        char_rnn_sentences = self.char_embedding_layer(sentences)
        # now we sum character and word embeddings
        if self.with_pretrained_word_embeddings:
            encoder_input = self.embeddings_dropout_layer(torch.concat((char_rnn_sentences, word_embeddings), dim=-1))
        else:
            encoder_input = self.embeddings_dropout_layer(char_rnn_sentences)
        # outputs = [batch, words, 2 * config.word_encoder_hidden_dim]
        outputs, _ = self.word_encoder(encoder_input)
        return self.encoder_dropout_layer(outputs)
