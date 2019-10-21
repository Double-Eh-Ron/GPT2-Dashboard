import json

class GPT2Config(object):
    def __init__(self, config_file):
        if 'distilgpt2-pytorch_model.bin' in config_file:
            self.json_file = 'GPT2/configs/gpt2_distil_config.json'
        if 'gpt2-pytorch_model.bin' in config_file:
            self.json_file = 'GPT2/configs/gpt2_small_config.json'
        elif 'gpt2-medium-pytorch_model.bin' in config_file:
            self.json_file = 'GPT2/configs/gpt2_medium_config.json'
        elif 'gpt2-large-pytorch_model.bin' in config_file:
            self.json_file = 'GPT2/configs/gpt2_large_config.json'
        else:
            self.json_file = 'GPT2/configs/gpt2_small_config.json'

        with open(self.json_file, 'r') as f:
            self.gpt2_setup = json.load(f)

        self.vocab_size = self.gpt2_setup['vocab_size']
        self.n_ctx = self.gpt2_setup['n_ctx']
        self.n_positions = self.gpt2_setup['n_positions']
        self.n_embd = self.gpt2_setup['n_embd']
        self.n_layer = self.gpt2_setup['n_layer']
        self.n_head = self.gpt2_setup['n_head']
        self.layer_norm_epsilon = self.gpt2_setup['layer_norm_epsilon']
        self.initializer_range = self.gpt2_setup['initializer_range']

    def output_config(self):
        return 'GPT-2 configuration used: ' + str(self.gpt2_setup)
