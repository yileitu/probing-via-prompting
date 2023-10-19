# -*- coding: utf-8 -*-
import json
import os

from run_pp import DataTrainingArguments

data_args = DataTrainingArguments()
data_args.data_dir = "./ontonotes/pp/"
data_args.task = "ner"
data_files = {
	"train"     : os.path.join(data_args.data_dir, data_args.task, 'train.json'),
	"validation": os.path.join(data_args.data_dir, data_args.task, 'development.json'),
	"test"      : os.path.join(data_args.data_dir, data_args.task, 'test.json')
	}


def count_data_in_json(file_path):
	with open(file_path, 'r') as f:
		lines = f.readlines()

	count = 0
	for line in lines:
		try:
			item = json.loads(line)
			count += 1
		except json.JSONDecodeError:
			print(f"Error decoding line: {line}")
			continue

	return count


counts = {}
total = 0
for key, file_path in data_files.items():
	counts[key] = count_data_in_json(file_path)
	total += counts[key]

percentages = {}
for key, count in counts.items():
	percentages[key] = (count / total) * 100

print("Counts:", counts)
print("Percentages:", percentages)

# raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
# raw_train: Dataset = raw_datasets["train"]
# raw_dev: Dataset = raw_datasets["validation"]
# raw_test: Dataset = raw_datasets["test"]
# token = "<|endoftext|>"
#
# labels = []
# for ex in raw_dev['text']:
# 	idx = ex.find(token) + len(token)
# 	labels.append(ex[idx:])
#
# labels_cnt = Counter(labels)
# labels_cnt_values = sorted(list(labels_cnt.values()), reverse=True)
#
# print(labels_cnt)
# print(labels_cnt_values)
# print("Mode pct: ", labels_cnt_values[0] / len(labels))
