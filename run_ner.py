# -*- coding: utf-8 -*-
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from allennlp.modules import scalar_mix
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import nn
from transformers import AdamW, AutoConfig, EarlyStoppingCallback, GPT2Model, GPT2PreTrainedModel, GPT2TokenizerFast, \
	HfArgumentParser, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, default_data_collator, \
	set_seed

from arguments import DataTrainingArguments, ModelArguments
from utils import record_num_of_params, setup_logger, setup_wandb

PADDING_LABEL_ID = -100


# def compute_metrics(eval_pred):
# 	output, labels = eval_pred
# 	print("output: ", output)
# 	print("labels: ", labels)
# 	# _, logits = output
# 	# predictions = np.argmax(logits, axis=-1)
# 	# metric = load_metric("accuracy")
# 	# return metric.compute(predictions=predictions, references=labels)\
# 	return None

def compute_metrics(eval_pred):
	logits, labels = eval_pred
	print("Dim of logits: ", logits.shape)
	print("Dim of labels: ", labels.shape)
	predictions = np.argmax(logits, axis=-1)

	# 将预测和真实标签的形状调整为一维
	true_labels = labels.flatten()
	pred_labels = predictions.flatten()

	# 过滤掉所有真实标签为 -100 的位置
	mask = true_labels != -100
	true_labels = true_labels[mask]
	pred_labels = pred_labels[mask]

	# 计算准确率
	accuracy = accuracy_score(true_labels, pred_labels)
	return {"accuracy": accuracy}


def preprocess_data(examples):
	# 对子词进行编码
	tokenized_inputs = tokenizer(
		examples["tokens"], is_split_into_words=True, padding="max_length", truncation=True, max_length=128
		)

	# 将标签与编码后的输入对齐
	labels = []
	for i, label_list in enumerate(examples["tags"]):
		word_ids = tokenized_inputs.word_ids(batch_index=i)
		label_ids = [PADDING_LABEL_ID if word_id is None else label_list[word_id] for word_id in word_ids]
		labels.append(label_ids)

	tokenized_inputs["labels"] = labels
	return tokenized_inputs


# def preprocess_data(examples):
# 	tokenized_inputs = {'input_ids': [], 'attention_mask': [], 'labels': []}
#
# 	for i in range(len(examples['tokens'])):
# 		tokens = examples['tokens'][i]
# 		label_ids = examples['tags'][i]
# 		input_ids = []
# 		attention_mask = []
# 		labels = []
#
# 		for token, label_id in zip(tokens, label_ids):
# 			# 使用 GPT-2 分词器对每个子词进行编码
# 			subwords = tokenizer.encode(token, add_special_tokens=False)
# 			input_ids.extend(subwords)
# 			attention_mask.extend([1] * len(subwords))
# 			# 使用相同的标签 ID 为每个子词标记
# 			labels.extend([label_id] * len(subwords))
#
# 		tokenized_inputs['input_ids'].append(input_ids)
# 		tokenized_inputs['attention_mask'].append(attention_mask)
# 		tokenized_inputs['labels'].append(labels)
#
# 	return tokenized_inputs


# def preprocess_data(example):
# 	tokens_list = example['tokens']
# 	tags_list = example['tags']
#
# 	input_ids = tokenizer.convert_tokens_to_ids(tokens_list)
# 	attention_masks = [1] * len(input_ids)
# 	labels = tags_list
#
# 	return {
# 		'input_ids'     : input_ids,
# 		'attention_mask': attention_masks,
# 		'labels'        : labels
# 		}


class GPT2ForNERWithProbe(GPT2PreTrainedModel):
	def __init__(self, config, gpt2):
		super().__init__(config)
		# Architecture
		self.gpt2 = gpt2
		self.scalar_mix = scalar_mix.ScalarMix(config.n_layer)

		# Trainable parameters
		if config.onehot is False:
			for param in self.gpt2.parameters():
				param.requires_grad = False
		else:
			for param in self.gpt2.parameters():
				param.requires_grad = True
			print("Onehot is True. All parameters are trainable.")

		# config
		self.model_parallel = False
		self.device_map = None
		self.num_labels = config.num_labels
		self.mlp_dim: int = config.mlp_dim
		self.mlp_layers: int = config.mlp_layers
		self.mlp_dropout = config.mlp_dropout
		self.use_mlp = config.use_mlp
		self.gpt2_hidden_size = config.hidden_size

		# Probe Architecture
		if self.use_mlp is False:
			# Linear Regression
			lin_module_list = []
			if self.mlp_layers == 1:
				self.probe = nn.Sequential(
					nn.Linear(self.gpt2_hidden_size, self.mlp_dim),
					nn.Linear(self.mlp_dim, self.num_labels)
					)
			elif self.mlp_layers >= 2:
				lin_module_list.append(nn.Linear(self.gpt2_hidden_size, self.mlp_dim))
				for _ in range(self.mlp_layers - 1):
					lin_module_list.append(nn.Linear(self.mlp_dim, self.mlp_dim))
				lin_module_list.append(nn.Linear(self.mlp_dim, self.num_labels))
				self.probe = nn.Sequential(*lin_module_list)
		else:
			# Multi-Layer Perceptron
			input_layer_list = [
				nn.Linear(self.gpt2_hidden_size, self.mlp_dim),
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
			self.probe = nn.Sequential(*classifier_module_list)

	def forward(
			self,
			input_ids=None,
			past_key_values=None,
			attention_mask=None,
			token_type_ids=None,
			position_ids=None,
			head_mask=None,
			inputs_embeds=None,
			encoder_hidden_states=None,
			encoder_attention_mask=None,
			labels=None,
			use_cache=None,
			output_attentions=None,
			return_dict=None,
			):

		gpt2_outputs = self.gpt2(
			input_ids,
			past_key_values=past_key_values,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
			position_ids=position_ids,
			head_mask=head_mask,
			inputs_embeds=inputs_embeds,
			encoder_hidden_states=encoder_hidden_states,
			encoder_attention_mask=encoder_attention_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
			output_hidden_states=True,
			return_dict=return_dict,
			)
		all_hidden_states = gpt2_outputs.hidden_states[1:]  # 不包括embedding层的输出
		contextual_embeddings = self.scalar_mix(all_hidden_states)
		logits = self.probe(contextual_embeddings)
		# 如果提供了标签，则计算损失，否则只返回logits
		output = (logits,)
		if labels is not None:
			loss_fn = torch.nn.CrossEntropyLoss()
			# 注意: 您可能需要调整标签的形状或应用mask以匹配logits的形状
			loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
			output = (loss,) + output

		return output


class SaveEvalResultsCallback(TrainerCallback):
	def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		global eval_results_df
		metrics = kwargs.pop("metrics")
		cur_epoch: int = int(state.epoch)

		if state.is_world_process_zero:
			eval_result = {
				"epoch"        : cur_epoch,
				"eval_accuracy": metrics["eval_accuracy"],
				"eval_loss"    : metrics["eval_loss"]
				}
			eval_result_df = pd.DataFrame([eval_result])
			eval_results_df = pd.concat([eval_results_df, eval_result_df])
			eval_results_df.to_csv(os.path.join(args.output_dir, f"eval_results.csv"), index=False)


if __name__ == '__main__':
	# Model arguments
	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	model_args, data_args, training_args = parser.parse_args_into_dataclasses()
	model_args: ModelArguments
	data_args: DataTrainingArguments
	training_args: TrainingArguments

	# Set up wandb
	serial = setup_wandb(training_args, model_args, data_args)

	# Set up other training arguments
	training_args.report_to = ["wandb"]
	training_args.run_name = serial
	training_args.logging_steps = 50
	training_args.load_best_model_at_end = True
	training_args.metric_for_best_model = "eval_accuracy"
	training_args.greater_is_better = True
	training_args.save_total_limit = 1

	# Miscellaneous
	logger = setup_logger(training_args)
	set_seed(training_args.seed)

	# Load gpt2
	gpt2 = GPT2Model.from_pretrained('gpt2')

	# Load tokenizer
	tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', add_prefix_space=True)
	gpt2.resize_token_embeddings(len(tokenizer))
	tokenizer.pad_token = tokenizer.eos_token
	pre_tokenizer = WhitespaceSplit()
	tokenizer.pre_tokenizer = pre_tokenizer

	# Load config for gpt2-probe model
	config = AutoConfig.from_pretrained('gpt2')
	config.num_labels = 37  # NOTE: 37 is the number of labels in tner/ontonotes dataset
	config.onehot = model_args.onehot
	if config.onehot:
		logger.info("Using onehot embeddings.")
	config.mlp_dropout = model_args.mlp_dropout
	config.mlp_dim = model_args.mlp_dim
	config.mlp_layers = model_args.mlp_layers
	config.use_mlp = model_args.use_mlp

	# Load gpt2-probe model
	model = GPT2ForNERWithProbe(config=config, gpt2=gpt2)
	# device = set_gpu_env(num_gpus=training_args.n_gpu)
	# model.to(device)
	record_num_of_params(model, logger)

	# Process the dataset
	raw_datasets = load_dataset("tner/ontonotes5")
	with training_args.main_process_first(desc="dataset map tokenization"):
		tokenized_datasets = raw_datasets.map(
			preprocess_data, batched=True
			)
	if training_args.do_train:
		if "train" not in tokenized_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = tokenized_datasets["train"]
	if training_args.do_eval:
		if "validation" not in tokenized_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = tokenized_datasets["validation"]

	# Optimizer
	if training_args.do_train:
		no_decay = ["bias", "LayerNorm.weight"]
		optimizer_grouped_parameters = [
			{
				"params"      : [p for n, p in model.named_parameters() if
				                 not any(nd in n for nd in no_decay) and p.requires_grad],
				"weight_decay": training_args.weight_decay,
				"lr"          : training_args.learning_rate
				},
			{
				"params"      : [p for n, p in model.named_parameters() if
				                 any(nd in n for nd in no_decay) and p.requires_grad],
				"weight_decay": 0.0,
				"lr"          : training_args.learning_rate
				},
			]
		optimizer = AdamW(optimizer_grouped_parameters)
	else:
		optimizer = None

	# Define a callback to save evaluation results in a csv file
	eval_results_df = pd.DataFrame(columns=["epoch", "eval_accuracy", "eval_loss"])

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		tokenizer=tokenizer,
		data_collator=default_data_collator,  # Data collator will default to DataCollatorWithPadding, so we change it.
		optimizers=(optimizer, None),
		compute_metrics=compute_metrics,
		callbacks=[SaveEvalResultsCallback(), EarlyStoppingCallback(early_stopping_patience=10)],
		)

	# Training
	if training_args.do_train:
		train_result = trainer.train()
		trainer.save_model(output_dir=training_args.output_dir)  # Saves the tokenizer too for easy upload
		metrics = train_result.metrics
		metrics["train_samples"] = len(train_dataset)
		logger.info(f"*** Train Metrics *** \n{metrics}")

	# Testing
	if training_args.do_eval:
		logger.info("*** Evaluate ***")
		logger.info(
			f'Layer weights: {torch.stack([p for n, p in model.scalar_mix.named_parameters() if "scalar" in n]).flatten()}'
			)
		metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
		metrics["eval_samples"] = len(eval_dataset)
		logger.info(f"*** Evaluate Metrics *** \n{metrics}")

	eval_results_df.to_csv(os.path.join(training_args.output_dir, "eval_results.csv"), index=False)
