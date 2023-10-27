from definitions import BOS_TOKEN, EOS_TOKEN


class Transforms:
    def __init__(self, vocab, vocab_words, vocab_chars):
        self.vocab = vocab
        self.vocab_words = vocab_words
        self.vocab_chars = vocab_chars

    def text_transform(self, word_list):
        words_chars = [[self.vocab_chars.vocab.get(ch, self.vocab_chars.vocab['<unk>']) for ch in word]
                       for word in word_list]
        words_fasttext = list()
        for word in word_list:
            words_fasttext.append(self.vocab_words.model[word])
        return words_chars, words_fasttext

    def label_transform_grams(self, labels, dataset_index):
        transformed = []
        for label in labels:
            transformed_label = [self.vocab.vocab[BOS_TOKEN]]
            for pos_gram in label.split():
                pos_gram = pos_gram + '_' + self.vocab.datasets_names[dataset_index]
                if pos_gram in self.vocab.vocab:
                    transformed_label.append(self.vocab.vocab[pos_gram])
            transformed_label.append(self.vocab.vocab[EOS_TOKEN])
            transformed.append(transformed_label)
        return transformed

    def label_transform_grams_multitask(self, labels, dataset_index):
        transformed = []
        for label in labels:
            transformed_word = [self.vocab.vocab[dataset_index][BOS_TOKEN]]
            for pos_gram in label.split():
                transformed_word.append(self.vocab.vocab[dataset_index][pos_gram])
            transformed_word.append(self.vocab.vocab[dataset_index][EOS_TOKEN])
            transformed.append(transformed_word)
        return transformed

    def label_transform_tags(self, labels, dataset_index):
        transformed = []
        for label in labels:
            transformed.append(self.vocab.vocab[dataset_index][label])
        return transformed
