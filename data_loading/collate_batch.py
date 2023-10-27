import torch
from data_loading.transforms import Transforms
from torch.nn.utils.rnn import pad_sequence
from torch.nn import ConstantPad1d
from definitions import PAD_INDEX
import numpy as np


class CollateBatch:
    def __init__(self, vocab, vocab_words, vocab_chars):
        self.vocab = vocab
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars
        self.transforms = Transforms(vocab, vocab_words, vocab_chars)

    def collate_text(self, _text, max_word_len):
        transformed_text = []
        words_embeddings = []
        for word, word_embedding in zip(*self.transforms.text_transform(_text)):
            transformed_text.append(torch.tensor(word))
            words_embeddings.append(word_embedding)
        transformed_text[0] = ConstantPad1d((0, max_word_len - len(transformed_text[0])), 0)(transformed_text[0])
        transformed_text = pad_sequence(transformed_text, batch_first=True)
        words_embeddings = torch.tensor(np.array(words_embeddings), dtype=torch.float)
        return transformed_text, words_embeddings

    def collate_labels_grams(self, _labels, max_grams_len, dataset_index):
        transformed_labels = []
        for label in self.transforms.label_transform_grams(_labels, dataset_index):
            transformed_labels.append(torch.tensor(label))
        transformed_labels[0] = ConstantPad1d((0, max_grams_len - len(transformed_labels[0])),
                                              PAD_INDEX)(transformed_labels[0])
        return pad_sequence(transformed_labels, batch_first=True)


    def collate_batch_grams(self, batch, batch_sampler):
        label_list, text_list, words_embeddings_list = [], [], []
        max_word_len = max(len(word) for entry in batch for word in entry[0])
        max_grams_len = max(len(grams.split()) for entry in batch for grams in entry[2]) + 2  # + (BOS, EOS)
        for (_text, _, _labels) in batch:
            transformed_text, words_embeddings = self.collate_text(_text, max_word_len)
            text_list.append(transformed_text)
            words_embeddings_list.append(words_embeddings)
            transformed_labels = self.collate_labels_grams(_labels, max_grams_len, batch_sampler.index)
            label_list.append(transformed_labels)
        return \
            batch_sampler.index, \
            pad_sequence(text_list, batch_first=True), \
            pad_sequence(words_embeddings_list, batch_first=True), \
            pad_sequence(label_list, batch_first=True).permute(0, 2, 1)  # to make it batch x grams x words
