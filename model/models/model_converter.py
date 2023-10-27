import torch
from model.layers.attention_layer import AttentionLayer
from definitions import BOS_INDEX, EOS_INDEX, PAD_INDEX, EOS_TOKEN
from model.models.model_base import ModelBase
import torch.nn as nn
from data_loading.collate_batch import CollateBatch
from data_loading.vocab import VocabGrams
from model.layers.decoder import ConverterWordDecoderRNN
from model.layers.label_embedding_layer import GramEmbeddingLayer
from torch.nn.utils.rnn import pad_sequence


class ModelConverter(ModelBase):
    def __init__(self, config):
        config['vocab'] = VocabGrams(config['datasets_names'])
        super().__init__(config)
        self.config = config
        self.config['collate_batch'] = CollateBatch(self.config['vocab'], self.config['vocab_words'],
                                                    self.config['vocab_chars'])
        self.config['collate_fn'] = self.config['collate_batch'].collate_batch_grams
        self.decoder = ConverterWordDecoderRNN(self.config)
        self.config['attention_nhead'] = len(self.config['datasets_names'])
        self.attention = AttentionLayer(self.config)
        self.label_embedding_layer = GramEmbeddingLayer(self.config)
        if self.config['weights_source_model'] is not None:
            self.encoder.load_state_dict(self.config['weights_source_model'].encoder.state_dict())
            if self.config['freeze_encoder']:
                self.encoder.requires_grad_(False)
            self.decoder.load_state_dict(self.config['weights_source_model'].decoder.state_dict())
            if self.config['freeze_decoder']:
                self.decoder.requires_grad_(False)
            self.label_embedding_layer.load_state_dict(
                self.config['weights_source_model'].label_embedding_layer.state_dict())
            if self.config['freeze_lel']:
                self.label_embedding_layer.requires_grad_(False)

    def forward(self, sentences, word_embeddings, dataset_index, input_labels, return_pseudo=True):
        # sentences = [batch size, words, max_word_len]
        # input_labels = [batch, max_grams_len, words]
        # word_embeddings = [batch, words, fasttext_dim]
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        words_transformer = self.encoder(sentences, word_embeddings)
        is_training = self._trainer is not None and self.trainer.training or self.training
        is_grad = torch.is_grad_enabled()
        if not self.config['pretrain']:
            self.train(mode=False)
            torch.set_grad_enabled(False)
        pseudo_outputs = list()
        for index in range(len(self.config['datasets_names'])):
            pseudo_outputs.append(torch.zeros(sentences.shape[0], sentences.shape[1],
                                  self.config['decoder_maximum_iterations'],
                                  len(self.config['vocab'].vocab)).to(words_transformer))
            self._decode(pseudo_outputs[-1], words_transformer, index, input_labels, False)
        if not self.config['pretrain']:
            self.train(mode=is_training)
            torch.set_grad_enabled(is_grad)
        if return_pseudo:
            return pseudo_outputs
        else:
            new_outputs = list()
            for index in range(len(self.config['datasets_names'])):
                if index != dataset_index:
                    pseudo_input_labels = pseudo_outputs[index].argmax(-1).permute(0, 2, 1)
                    new_outputs.append(torch.zeros(sentences.shape[0], sentences.shape[1],
                                                   input_labels.shape[1],
                                                   len(self.config['vocab'].vocab)).to(words_transformer))
                    self._decode(new_outputs[-1], words_transformer, dataset_index, pseudo_input_labels,
                                 True, input_labels)
            return torch.cat(new_outputs, dim=0)

    def _decode(self, outputs, words_transformer, dataset_index, input_labels, is_generated, target=None):
        max_sentence_len = input_labels.shape[2]
        batch_size = input_labels.shape[0]
        # we decrease by one to reserve last element to be <eos> in case it was not generated
        max_grams_len = outputs.shape[-2]
        if target is None:
            max_grams_len -= 1
        mask = self.generate_attention_mask(input_labels, is_generated)
        # merge batch and sentence_len to process all words in all bathces in one go
        inp = input_labels[:, 0, :].reshape(-1)
        outputs[:, :, 0, BOS_INDEX] = 1
        hidden = words_transformer.reshape(-1, words_transformer.shape[2])
        target_lel = self.get_target_embeddings_from_lel(input_labels)
        # for each gram, excluding <BOS>
        for j in range(1, max_grams_len):
            output = self.decoder_step(inp, hidden, target_lel, mask)
            self._mask_predictions(output, dataset_index)
            # place predictions in a tensor holding predictions for each token
            outputs[:, :, j, :] = output.view(batch_size, max_sentence_len, -1)

            if target is not None and (self._trainer is not None and self.trainer.training or self.training):
                inp = target[:, j, :].reshape(-1)
            else:
                inp = output.argmax(-1)
        else:
            if target is None:
                outputs[:, :, max_grams_len, EOS_INDEX] = 1

    def decoder_step(self, inp, hidden, target_lel, mask):
        hidden = self.decoder(inp, hidden)
        output = self.label_embedding_layer(self.attention(hidden, target_lel, mask))
        return output

    def generate_attention_mask(self, input_labels, is_generated):
        # target = [batch, max_grams_len, words]

        if not is_generated:
            mask = torch.eq(input_labels.permute(0, 2, 1).reshape(-1, input_labels.shape[1]), PAD_INDEX)
        else:
            # since we use dual learning, input labels have grammemes (or <pad>) after <eos>, so we have to mask it
            # to do so, we first find the position of first <eos>
            eos_pos = torch.isin(input_labels.permute(0, 2, 1).reshape(-1, input_labels.shape[1]), EOS_INDEX).to(
                torch.float).argmax(dim=-1, keepdim=True)

            # then we create mask with True _after_ <eos> because that is expected in attention layer
            indices = torch.arange(start=0, end=input_labels.shape[1], step=1).to(self.device)
            indices = indices.expand(eos_pos.shape[0], indices.shape[0])
            mask = torch.gt(indices, eos_pos)
            # replace grammemes after <eos> with <pad>> for input labels (for test phase)
            padding_mask = mask.view(input_labels.shape[0], input_labels.shape[2], input_labels.shape[1]).permute(
                0, 2, 1)
            input_labels.masked_fill_(padding_mask, PAD_INDEX)

        # mask = [batch * words, max_grams_len]
        return mask

    def get_target_embeddings_from_lel(self, target):
        # target = [batch, max_grams_len, words]
        # hidden = [batch * words, 2 * config['word_encoder_hidden_dim]

        # get target embeddings from LEL to use it in attention
        # target_one_hot = [batch * words, num_classes, grams]
        target_one_hot = nn.functional.one_hot(target.permute(0, 2, 1).reshape(-1, target.shape[1]),
                                               num_classes=len(self.config['vocab'].vocab)).permute(0, 2, 1)
        # return = [batch * words, max_grams_len, hidden_dim]
        # decoder.out.linear.weight = [grams, hidden_dim] because linear layer does double transpose
        return torch.matmul(input=self.label_embedding_layer.linear.weight.transpose(0, 1),
                            other=target_one_hot.to(torch.float)).permute(0, 2, 1)

    def predict(self, words_list, input_labels, dataset_index_from, dataset_index_to):
        self.config['vocab_words'].load_fasttext_model()
        self.train(mode=False)
        with torch.no_grad():
            transformed_text, words_embeddings = self.config['collate_batch'].collate_text(words_list, len(words_list))
            transformed_text = pad_sequence([transformed_text], batch_first=True).to(self.device)
            words_embeddings = pad_sequence([words_embeddings], batch_first=True).to(self.device)
            transformed_labels = pad_sequence([self.config['collate_batch'].collate_labels_grams(
                input_labels, max(len(label.split(' ')) for label in input_labels), dataset_index_from)],
                                              batch_first=True).permute(0, 2, 1).to(self.device)
            predictions = self(
                transformed_text, words_embeddings, dataset_index_from, transformed_labels)[dataset_index_to]
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
