from configs.base_config import config

config['weights_source_model'] = None
config['pretrain'] = False
config['freeze_encoder'] = False
config['freeze_decoder'] = True
config['freeze_lel'] = False
config['lr'] = 1.1

# attention
config['attention_dropout'] = 0.1

# testing
config['batch_size_test'] = 1

config['datasets_names'] = []

config['experiment_name'] = 'model_converter'
config['version_name'] = None
