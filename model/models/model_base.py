import torch
from model.layers.encoder import ConverterEncoder
import pytorch_lightning as pl
from data_loading.vocab import VocabWords, VocabChars


class ModelBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config['vocab_words'] = VocabWords(config['datasets_names'], config['fasttext_path'])
        self.config['vocab_chars'] = VocabChars(config['datasets_names'])
        self.config['word_encoder_nhead'] = len(self.config['datasets_names'])
        self.encoder = ConverterEncoder(config)
        torch.set_float32_matmul_precision('high')

    def _encode(self, sentences, word_embeddings):
        # sentences = [batch size, words, max_word_len]
        # word_embeddings = [batch, words, fasttext_dim]
        hidden_all = self.encoder(sentences, word_embeddings)
        return hidden_all

    def _mask_predictions(self, predictions, dataset_index, inplace=True):
        if inplace:
            predictions.index_fill_(-1, self.config['vocab'].dataset_to_non_vocab_indices[dataset_index].to(
                self.device), float('-inf'))
        else:
            return predictions.index_fill(-1, self.config['vocab'].dataset_to_non_vocab_indices[dataset_index].to(
                self.device), float('-inf'))
