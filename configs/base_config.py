config = {
    # Input
    'word_embeddings_dropout': 0.25,
    # words
    'word_embedding_dim': 300,
    'fasttext_path': 'resources/fasttext/cc.ru.300.bin',
    'with_pretrained_word_embeddings': True,

    # character_encoder
    'char_rnn_layers_num': 1,
    'char_rnn_hidden_dim': 50,  # lstm on chars
    'char_rnn_bidirectional': True,
    'char_embedding_dim': 50,

    # encoder
    'word_encoder_dropout': 0,
    'word_encoder_num_layers': 1,
    'word_encoder_bidirectional': True,

    # decoder (default; special values in corresponding config if necessary)
    'decoder_dropout': 0,
    'decoder_embeddings_dropout': 0.25,
    'decoder_maximum_iterations': 17,

    # training
    'batch_size_words': 64,
    'batch_size_words_val': 512,
    'lr': 2,
    'weight_decay': 0,
    'eps': 1e-8,

    'nepoch_no_imprv': 25,
    'max_epochs': 10
}

config['char_rnn_output_dim'] = config['char_rnn_hidden_dim'] * (int(config['char_rnn_bidirectional']) + 1)

# encoder
config['word_encoder_input_dim'] = config['char_rnn_output_dim'] + \
                                   config['word_embedding_dim'] * int(config['with_pretrained_word_embeddings'])
