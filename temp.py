# -*- coding: utf-8 -*-
import pickle
from typing import Dict, List, Tuple

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from utils import LABEL_DICT, word_to_one_hot

label2id = {label: i for i, label in enumerate(LABEL_DICT["ner"])}
num_labels = len(label2id)

# data_files = {"train": "toy_dataset.json"}
# raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
# raw_train: Dataset = raw_datasets["train"]
#
# label2id = {label: i for i, label in enumerate(LABEL_DICT["ner"])}
# num_labels = len(label2id)
#
# # Define batch size for processing data in batches
# batch_size = 128
#
#
# # Define function to process a batch of data and return a TensorDataset
# def process_batch(batch_data: List[Tuple[str, List[Dict[str, int]]]]) -> TensorDataset:
# 	targets_list = []
# 	labels_list = []
# 	for datapoint in batch_data:
# 		text = datapoint[0]
# 		targets = datapoint[1]
# 		for target in targets:
# 			start, end = target["span1"]
# 			target_word = text.split(" ")[start:end + 1]  # Extract the target word from the text
# 			targets_list.append(target_word)
# 			label = label2id[target["label"]]
# 			labels_list.append(label)
#
# 	# Convert each target to a one-hot encoding matrix and store it in a PyTorch tensor
# 	one_hot_targets = torch.zeros(len(targets_list), len(targets_list))
# 	for i, target in enumerate(targets_list):
# 		one_hot_targets[i] = word_to_one_hot(word=target, vocab=targets_list)
#
# 	y = torch.tensor(labels_list, dtype=torch.long)
# 	# Create TensorDataset for X and y
# 	dataset = TensorDataset(one_hot_targets, y)
# 	return dataset
#
#
# # Create DataLoader for processing data in batches
# dataloader = torch.utils.data.DataLoader(raw_train, batch_size=batch_size)
# # Save the TensorDataset to a file using pickle
# with open('onehot_dataset.pkl', 'wb') as f:
# 	for batch_data in dataloader:
# 		dataset = process_batch(batch_data)
# 		pickle.dump(dataset, f)

# Load the JSON dataset
data_files = {"train": "toy_dataset.json"}
raw_datasets: DatasetDict = load_dataset("json", data_files=data_files)
raw_train: Dataset = raw_datasets["train"]

# # Iterate over the datapoints in the dataset and extract the targets
# targets_list = []
# labels_list = []
# for datapoint in raw_train:
# 	text = datapoint["text"]
# 	targets = datapoint["targets"]
# 	for target in targets:
# 		start, end = target["span1"]
# 		target_word = text.split(" ")[start:end + 1]  # Extract the target word from the text
# 		targets_list.append(target_word)
# 		label = label2id[target["label"]]
# 		labels_list.append(label)
#
# # Convert each target to a one-hot encoding matrix and store it in a list
# one_hot_targets = []
# for target in targets_list:
# 	one_hot = word_to_one_hot(word=target, vocab=targets_list)
# 	one_hot_targets.append(one_hot)

# Iterate over the datapoints in the dataset and extract the labels corresponding to each target
labels_list = []
for datapoint in raw_train:
	text = datapoint["text"]
	targets = datapoint["targets"]
	for target in targets:
		label = label2id[target["label"]]
		labels_list.append(label)

# Convert the list of one-hot encodings to a PyTorch tensor
X: Tensor = torch.eye(len(labels_list))
y = torch.tensor(labels_list, dtype=torch.long)
print(X)
print(y)

# # Convert the list of one-hot encodings to a PyTorch tensor
# X: Tensor = torch.stack(one_hot_targets)
# print(X)
# y = torch.tensor(labels_list, dtype=torch.long)
# print(y)
# Create DataLoader for X and y
dataset = TensorDataset(X, y)
# Save the TensorDataset to a file using pickle
with open('onehot_dataset.pkl', 'wb') as f:
	pickle.dump(dataset, f)


# with open('onehot_dataset.pkl', 'rb') as handle:
# 	dataset: TensorDataset = pickle.load(handle)
#
# input_dim = dataset.tensors[0].shape[1]
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# mlp = OneHotMlpProbe(input_dim=input_dim, num_labels=num_labels, mlp_layers=8, mlp_dim=512, mlp_dropout=0.0)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(mlp.parameters(), lr=0.001)
#
# # Check if a GPU is available
# if torch.cuda.is_available():
# 	device = torch.device('cuda')
# else:
# 	device = torch.device('cpu')
#
# # Train the neural network
# num_epochs = 10
# for epoch in range(num_epochs):
# 	epoch_loss = 0
# 	correct = 0
# 	total = 0
# 	for batch_X, batch_y in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False):
# 		# Move batch_X and batch_y to device
# 		batch_X = batch_X.to(device)
# 		batch_y = batch_y.to(device)
#
# 		# Forward pass
# 		outputs = mlp(batch_X)
# 		loss = criterion(outputs, batch_y)
#
# 		# Backward pass and optimize
# 		optimizer.zero_grad()
# 		loss.backward()
# 		optimizer.step()
#
# 		# Calculate accuracy
# 		_, predicted = torch.max(outputs.data, 1)
# 		total += batch_y.size(0)
# 		correct += (predicted == batch_y).sum().item()
#
# 		# Update epoch loss
# 		epoch_loss += loss.item()
#
# 	# Print epoch statistics
# 	accuracy = correct / total
# 	print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
