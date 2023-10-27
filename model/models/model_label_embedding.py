import torch
import torch.nn as nn
from definitions import PAD_INDEX, BOS_INDEX, EOS_INDEX, EOS_TOKEN
from model.layers.decoder import ConverterWordDecoderRNN
from model.models.model_base import ModelBase
from data_loading.vocab import VocabGrams
from model.layers.label_embedding_layer import GramEmbeddingLayer
from data_loading.collate_batch import CollateBatch
from torch.nn.utils.rnn import pad_sequence


class ModelLabelEmbedding(ModelBase):
    def __init__(self, config):
        config['vocab'] = VocabGrams(config['datasets_names'])
        super().__init__(config)
        self.config = config
        self.config['collate_batch'] = CollateBatch(self.config['vocab'], self.config['vocab_words'],
                                                    self.config['vocab_chars'])
        self.config['collate_fn'] = self.config['collate_batch'].collate_batch_grams
        self.decoder = ConverterWordDecoderRNN(self.config)
        self.label_embedding_layer = GramEmbeddingLayer(self.config)

    def forward(self, sentences, word_embeddings, dataset_index, target=None):

        # sentences = [batch size, words, max_word_len]
        # target = [batch, max_grams_len, words]
        # word_embeddings = [batch, words, fasttext_dim]

        max_grams_len = target.shape[1] if target is not None else self.config['decoder_maximum_iterations']

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        words_transformer = self.encoder(sentences, word_embeddings)

        # tensor to store decoder outputs
        outputs = torch.zeros(sentences.shape[0], sentences.shape[1], max_grams_len,
                              len(self.config['vocab'].vocab)).to(words_transformer)
        self._decode(outputs, words_transformer, dataset_index, target)
        return outputs

    def _decode(self, outputs, words_transformer, dataset_index, target=None):
        # merge batch and sentence_len to process all words in all bathces in one go
        inp = torch.full((outputs.shape[0] * outputs.shape[1],), BOS_INDEX, device=self.device)
        outputs[:, :, 0, BOS_INDEX] = 1
        hidden = words_transformer.reshape(-1, words_transformer.shape[2])
        # for each gram, excluding <BOS>
        for j in range(1, outputs.shape[2]):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            hidden = self.decoder(inp, hidden)
            output = self.label_embedding_layer(hidden)
            self._mask_predictions(output, dataset_index)
            # place predictions in a tensor holding predictions for each token
            outputs[:, :, j, :] = output.view(outputs.shape[0], outputs.shape[1], -1)
            # we use teacher forcing only when training
            if self._trainer is not None and self.trainer.training or self.training:
                inp = target[:, j, :].view(-1)
            else:
                inp = output.argmax(-1)
        return outputs

    def predict(self, words_list, dataset_index):
        self.config['vocab_words'].load_fasttext_model()
        self.train(mode=False)
        with torch.no_grad():
            transformed_text, words_embeddings = self.config['collate_batch'].collate_text(words_list, len(words_list))
            transformed_text = pad_sequence([transformed_text], batch_first=True).to(self.device)
            words_embeddings = pad_sequence([words_embeddings], batch_first=True).to(self.device)
            predictions = self(transformed_text, words_embeddings, dataset_index)
            predictions = torch.argmax(predictions, dim=-1)
            labels = []
            for i in range(predictions.shape[1]):
                grams = []
                for j in range(1, predictions.shape[2]):
                    gram = self.config['vocab'].reversed_vocab[predictions[0][i][j].item()]
                    if gram == EOS_TOKEN:
                        break
                    grams.append(gram)
                labels.append(' '.join(grams))
            return labels
