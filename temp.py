from pprint import pprint
from dataclasses import dataclass, fields
from transformers import AutoConfig, GPT2Config, GPT2LMHeadModel

config = AutoConfig.from_pretrained("gpt2")
config.init_mean = 0.2
config.init_std = 0.2

pprint(type(config))
pprint(config)
