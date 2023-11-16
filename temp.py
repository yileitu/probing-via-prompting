# -*- coding: utf-8 -*-
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
print("Is this tokenizer fast (Rust-based)?:", tokenizer.is_fast)
