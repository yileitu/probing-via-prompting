# -*- coding: utf-8 -*-
import os
from copy import deepcopy

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, load_dataset
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from run_dp import DataTrainingArguments
from utils import LABEL_DICT


class OneHotMlpProbe(nn.Module):
	def __init__(self, input_dim: int, num_labels: int, mlp_layers: int = 1, mlp_dim: int = 256,
	             mlp_dropout: float = 0.1):
		super().__init__()
		self.input_dim = input_dim
		self.mlp_layers = mlp_layers
		self.mlp_dim = mlp_dim
		self.mlp_dropout = mlp_dropout
		self.num_labels = num_labels

		input_layer_list = [
			nn.Linear(self.input_dim, self.mlp_dim),
			nn.Tanh(),
			nn.LayerNorm(self.mlp_dim),
			nn.Dropout(self.mlp_dropout),
			]
		output_layer_list = [nn.Linear(self.mlp_dim, self.num_labels)]
		if self.mlp_layers == 1:
			classifier_module_list = deepcopy(input_layer_list) + deepcopy(output_layer_list)
		elif self.mlp_layers >= 2:
			classifier_module_list = deepcopy(input_layer_list)
			for _ in range(self.mlp_layers - 1):
				classifier_module_list.append(nn.Linear(self.mlp_dim, self.mlp_dim))
				classifier_module_list.append(nn.Tanh())
				classifier_module_list.append(nn.LayerNorm(self.mlp_dim))
				classifier_module_list.append(nn.Dropout(self.mlp_dropout))
			classifier_module_list += deepcopy(output_layer_list)
		else:
			raise ValueError(f"The num of MLP layers should be a positive integer. Your input is {self.mlp_layer}")
		self.classifier = nn.Sequential(*classifier_module_list)

	def forward(self, x):
		return self.classifier(x)

def tokenize_function(example):
	result = tokenizer(example['text'], padding="max_length", max_length=MAX_LENGTH[data_args.task])
	pre_tokenized_str = pre_tokenizer.pre_tokenize_str(example['text'])

	num_targets = len(example['targets'])
	num_to_pad = MAX_TARGET[data_args.task] - num_targets
	pad_spans = [[-1, -1]] * num_to_pad
	pad_labels = [-1] * num_to_pad

	result['span1s'] = [convert_span(result, pre_tokenized_str, target['span1']) for target in example['targets']]
	result['span1s'].extend(pad_spans)
	result['labels'] = [label2id[target['label']] for target in example['targets']]
	result['labels'].extend(pad_labels)
	if not config.unary:
		result['span2s'] = [convert_span(result, pre_tokenized_str, target['span2']) for target in
		                    example['targets']]
		result['span2s'].extend(pad_spans)
	return result


if __name__ == "__main__":
	# parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	# model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	wandb_proj_name = f"Onehot-Probe"
	serial = wandb.util.generate_id()

	os.environ["WANDB_PROJECT"] = wandb_proj_name
	wandb.init(
		project=wandb_proj_name,
		name=serial,
		)

	data_args = DataTrainingArguments()
	data_args.data_dir = "./ontonotes/dp/"
	data_args.task = "ner"

	data_files = {
		"train"     : os.path.join(data_args.data_dir, data_args.task, 'train.json'),
		# "validation": os.path.join(data_args.data_dir, data_args.task, 'development.json'),
		# "test"      : os.path.join(data_args.data_dir, data_args.task, 'test.json'),
		}
	raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
	raw_train: Dataset = raw_datasets["train"]

	tokenizer.pad_token = tokenizer.eos_token
	pre_tokenizer = WhitespaceSplit()
	tokenizer.pre_tokenizer = pre_tokenizer


	label2id = {label: i for i, label in enumerate(LABEL_DICT[data_args.task])}
	num_labels = len(label2id)


	mlp = OneHotMlpProbe(input_dim=X.shape[1], num_labels=num_labels, mlp_layers=8, mlp_dim=512, mlp_dropout=0.0)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(mlp.parameters(), lr=0.001)

	# Check if a GPU is available
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Train the neural network
	num_epochs = 10
	for epoch in range(num_epochs):
		epoch_loss = 0
		correct = 0
		total = 0
		for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
			# Move batch_X and batch_y to device
			batch_X = batch_X.to(device)
			batch_y = batch_y.to(device)

			# Forward pass
			outputs = mlp(batch_X)
			loss = criterion(outputs, batch_y)

			# Backward pass and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Calculate accuracy
			_, predicted = torch.max(outputs.data, 1)
			total += batch_y.size(0)
			correct += (predicted == batch_y).sum().item()

			# Update epoch loss
			epoch_loss += loss.item()

		# Print epoch statistics
		accuracy = correct / total
		print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
