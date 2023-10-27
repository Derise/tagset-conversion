from definitions import (BOS_INDEX, BOS_TOKEN, EOS_INDEX, EOS_TOKEN, PAD_INDEX, PAD_TOKEN, SNYATNIK_DATASET_DIR,
                         SYNTAGRUS_DATASET_DIR, SYNTAGRUS_DIALOGUE_DATASET_DIR, RNC_DIALOGUE_DATASET_DIR,
                         OPENCORPORA_DIALOGUE_DATASET_DIR, GIKRYA_DIALOGUE_DATASET_DIR)
import torch
import pickle
import fasttext
import pathlib

DATASET_TO_PATH = {"snyatnik": SNYATNIK_DATASET_DIR,
                   "syntagrus": SYNTAGRUS_DATASET_DIR,
                   "syntagrus_dialogue": SYNTAGRUS_DIALOGUE_DATASET_DIR,
                   "rnc_dialogue": RNC_DIALOGUE_DATASET_DIR,
                   "gikrya_dialogue": GIKRYA_DIALOGUE_DATASET_DIR,
                   "opencorpora_dialogue": OPENCORPORA_DIALOGUE_DATASET_DIR}


class VocabChars:
    def __init__(self, datasets_names):
        self.datasets_names = datasets_names
        self.vocab = self.get_chars_vocab()
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}

    def get_chars_vocab(self):
        vocab = dict()
        for dataset_name in self.datasets_names:
            vocab_path = DATASET_TO_PATH[dataset_name] / 'chars.vocab'
            if not vocab_path.exists():
                raise ValueError(f'{vocab_path} should be a valid path to chars vocab')
            with open(vocab_path, 'rb') as f:
                vocab.update(pickle.load(f))
        return vocab


class VocabWords:
    def __init__(self, datasets_names, model_path):
        self.datasets_names = datasets_names
        self.model_path = model_path
        if not pathlib.Path(self.model_path).exists():
            raise ValueError(f'{self.model_path} should be a valid path to a fastText model')
        self.model = None
        self.load_fasttext_model()

    def load_fasttext_model(self):
        if self.model is None:
            self.model = fasttext.load_model(self.model_path)


class VocabGrams:
    def __init__(self, datasets_names):
        self.datasets_names = datasets_names
        self.dataset_to_vocab_indices = dict()
        self.dataset_to_non_vocab_indices = dict()
        self.vocab, self.reversed_vocab = self.get_grams_vocab()
        self.get_non_vocab_indices_per_dataset()

    def get_grams_vocab(self):
        vocab = {PAD_TOKEN: PAD_INDEX, BOS_TOKEN: BOS_INDEX, EOS_TOKEN: EOS_INDEX}
        reversed_vocab = {PAD_INDEX: PAD_TOKEN, BOS_INDEX: BOS_TOKEN, EOS_INDEX: EOS_TOKEN}
        for i, dataset_name in enumerate(self.datasets_names):
            vocab_path = DATASET_TO_PATH[dataset_name] / 'grams.vocab'
            if not vocab_path.exists():
                raise ValueError(f'{vocab_path} should be a valid path to grams vocab')
            with open(vocab_path, 'rb') as f:
                dataset_vocab = pickle.load(f)
            self.dataset_to_vocab_indices[i] = [PAD_INDEX, BOS_INDEX, EOS_INDEX]
            for index, grammeme in enumerate(list(dataset_vocab.keys())[3:], start=len(vocab)):
                vocab[grammeme + '_' + dataset_name] = index
                reversed_vocab[index] = grammeme
                self.dataset_to_vocab_indices[i].append(index)
        return vocab, reversed_vocab

    def get_non_vocab_indices_per_dataset(self):
        for dataset_index in self.dataset_to_vocab_indices:
            self.dataset_to_non_vocab_indices[dataset_index] = torch.tensor(
                [i for i in self.vocab.values() if i not in self.dataset_to_vocab_indices[dataset_index]],
                dtype=torch.int64)
