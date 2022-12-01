# -*- coding: utf-8 -*-
import os
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

from run_pp import DataTrainingArguments

data_args = DataTrainingArguments()
data_args.data_dir = "./ontonotes/pp/"
data_args.task = "ner"
data_files = {
	"train"     : os.path.join(data_args.data_dir, data_args.task, 'train.json'),
	"validation": os.path.join(data_args.data_dir, data_args.task, 'development.json'),
	"test"      : os.path.join(data_args.data_dir, data_args.task, 'test.json')
	}
raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
raw_train: Dataset = raw_datasets["train"]
raw_dev: Dataset = raw_datasets["validation"]
raw_test: Dataset = raw_datasets["test"]
token = "<|endoftext|>"

labels = []
for ex in raw_dev['text']:
	idx = ex.find(token) + len(token)
	labels.append(ex[idx:])

labels_cnt = Counter(labels)
labels_cnt_values = sorted(list(labels_cnt.values()), reverse=True)

print(labels_cnt)
print(labels_cnt_values)
print("Mode pct: ", labels_cnt_values[0] / len(labels))
