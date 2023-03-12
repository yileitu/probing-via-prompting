# -*- coding: utf-8 -*-
from typing import List

from datasets import DatasetDict, load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

from run_dp import DataTrainingArguments
from utils import LABEL_DICT

tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
pre_tokenizer = WhitespaceSplit()
tokenizer.pre_tokenizer = pre_tokenizer

text = 'This is an example sentence'
encoded_text = tokenizer.encode(text, add_special_tokens=False)
one_hot = [0] * tokenizer.vocab_size
print(encoded_text)
print(tokenizer.vocab_size)
for token in encoded_text:
	one_hot[token] = 1

data_args = DataTrainingArguments()
data_args.data_dir = "./ontonotes/dp/"
data_args.task = "ner"

data_files = {
	"train": "toy_dataset.json",
	# "train"     : os.path.join(data_args.data_dir, data_args.task, 'train.json'),
	# "validation": os.path.join(data_args.data_dir, data_args.task, 'development.json'),
	# "test"      : os.path.join(data_args.data_dir, data_args.task, 'test.json'),
	}
raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
column_names: List[str] = raw_datasets["train"].column_names
print(column_names)

label2id = {label: i for i, label in enumerate(LABEL_DICT[data_args.task])}
num_labels = len(label2id)


def convert_span(result, pre_tokenized_str, span):
	char_start = pre_tokenized_str[span[0]][1][0]
	char_end = pre_tokenized_str[span[1]][1][1] - 1
	start = result.char_to_token(char_start)
	end = result.char_to_token(char_end)
	return [start, end]


def tokenize_function(example):
	result = tokenizer(example['text'])
	# tokenized_ids = result['input_ids'].copy()
	# result['input_ids'] = [0] * tokenizer.vocab_size
	# for token in tokenized_ids:
	# 	result['input_ids'][token] = 1

	pre_tokenized_str = pre_tokenizer.pre_tokenize_str(example['text'])

	result['span1s'] = [convert_span(result, pre_tokenized_str, target['span1']) for target in example['targets']]
	result['labels'] = [label2id[target['label']] for target in example['targets']]

	return result


tokenized_datasets = raw_datasets.map(
	tokenize_function, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names,
	load_from_cache_file=not data_args.overwrite_cache, desc="Running tokenizer on dataset"
	)
print(tokenized_datasets["train"][0])
