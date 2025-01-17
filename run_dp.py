# -*- coding: utf-8 -*-
import os
import random
import sys
from dataclasses import asdict

import pandas as pd
import torch
from datasets import load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AdamW, AutoConfig, AutoTokenizer, BertTokenizerFast, EarlyStoppingCallback, \
	GPT2LMHeadModel, \
	HfArgumentParser, \
	Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import wandb
from dp_arguments import DataTrainingArguments, ModelArguments
from modeling_gated_gpt2 import GPT2Model
from modeling_gpt2_dp import GPT2ForDiagnosticProbing
from utils import LABEL_DICT, convert_gate_to_mask, record_num_of_params, set_gpu_env, setup_logger, transform_dict

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MAX_LENGTH = {'pos': 350, 'const': 350, 'ner': 350, 'coref': 280, 'srl': 350}
MAX_TARGET = {'pos': 275, 'const': 175, 'ner': 71, 'coref': 300, 'srl': 11}
IS_UNARY = {'pos': True, 'const': True, 'ner': True, 'coref': False, 'srl': False}

GPT2_ZH_PATH = "uer/gpt2-chinese-cluecorpussmall"

# Define a callback to save evaluation results in a csv file
eval_results_df = pd.DataFrame(columns=["epoch", "eval_accuracy", "eval_loss"])


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


def main():
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
	else:
		model_args, data_args, training_args = parser.parse_args_into_dataclasses()

	# # Post-processing
	# GPT-2 English or Chinese:
	if model_args.chinese:
		model_args.gpt2_name_or_path = GPT2_ZH_PATH
		model_args.config_name = GPT2_ZH_PATH
		model_args.tokenizer_name = GPT2_ZH_PATH

	# Randomized
	if model_args.randomized or model_args.mod_randomized or model_args.agg_mod_rand or model_args.fine_mod_rand \
			or model_args.norm_mod_rand:
		model_args.gpt2_name_or_path = None
		model_args.config_name = "gpt2"
		model_args.tokenizer_name = "gpt2"
	# Determine the default experiment serial
	serial = f"Epoch{int(training_args.num_train_epochs)}-LR{training_args.learning_rate}-"
	if model_args.randomized:
		serial += "Randomized-"
	elif model_args.mod_randomized:
		serial += "ModRand-"
	else:
		serial += "Pretrained-"
	if model_args.dev:
		serial += "Dev"
	else:
		serial += "Test"

	# WanDB setup
	if model_args.use_mlp:
		wandb_proj_name = f"ConvergedProbe-{data_args.task}-DPMLP-Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}"
	else:
		wandb_proj_name = f"ConvergedProbe-{data_args.task}-DPLR-Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}"

	if model_args.mod_randomized:
		wandb_proj_name += "-ModRand"

	if model_args.onehot:
		wandb_proj_name += "-OneHot"

	if model_args.chinese:
		wandb_proj_name += "-Chinese"
		training_args.output_dir += "Chinese/"

	# CONCERN: 写得不优美，先用verbose代替处理如何控制wandb分组
	if model_args.verbose == 1 and model_args.mod_randomized:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-ModRand-Mean{model_args.init_mean}-Std{model_args.init_std}"
		serial = f"LR{training_args.learning_rate}-ModRand"

	if model_args.verbose == 2 and model_args.saturated:
		if model_args.randomized:
			wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-Saturated-Randomized"
		elif model_args.mod_randomized:
			wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-Saturated-ModRand"
		else:
			wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-Saturated-Pretrained"
		group_name = f"Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}-Epoch{int(training_args.num_train_epochs)}"
		serial = f"LR{training_args.learning_rate}-Saturated"

	if model_args.agg_mod_rand:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-AggModRand-Normal"
		group_name = f"Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}-Epoch{int(training_args.num_train_epochs)}"
		serial = f"LR{training_args.learning_rate}-AggModRand"
	if model_args.fine_mod_rand:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-FineModRand-Normal"
		group_name = f"Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}-Epoch{int(training_args.num_train_epochs)}"
		serial = f"LR{training_args.learning_rate}-FineModRand"
	if model_args.norm_mod_rand:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-NormModRand"
		group_name = f"Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}-Epoch{int(training_args.num_train_epochs)}"
		serial = f"LR{training_args.learning_rate}-NormModRand"
	serial += f"-Seed{training_args.seed}"

	os.environ["WANDB_PROJECT"] = wandb_proj_name
	wandb.init(
		project=wandb_proj_name,
		name=serial,
		)

	# Set up training arguments
	training_args.report_to = ["wandb"]
	training_args.logging_steps = 50
	training_args.run_name = serial
	training_args.load_best_model_at_end = True
	training_args.metric_for_best_model = "eval_accuracy"
	training_args.greater_is_better = True
	training_args.save_total_limit = 1

	wandb.log(transform_dict(asdict(model_args)))
	wandb.log(transform_dict(asdict(data_args)))

	# Misc Setup
	set_seed(training_args.seed)
	logger = setup_logger(training_args)
	device = set_gpu_env(num_gpus=model_args.n_gpu)

	# Detecting last checkpoint.
	last_checkpoint = None
	if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
				)
		elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
				)

	data_files = {}
	dataset_args = {}
	logger.info("Loading data for {}".format(data_args.task))
	if training_args.do_train:
		data_files["train"] = os.path.join(data_args.data_dir, data_args.task, 'train.json')

	if model_args.dev:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'development.json')
	else:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')
	data_files["test"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')

	raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
	if "_control" in data_args.task:
		data_args.task = data_args.task.replace("_control", "")
	label2id = {label: i for i, label in enumerate(LABEL_DICT[data_args.task])}

	# Load GPT2 config
	config_kwargs = {
		"cache_dir"     : model_args.cache_dir,
		"revision"      : model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
		}
	if model_args.config_name:
		config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
	elif model_args.gpt2_name_or_path:
		config = AutoConfig.from_pretrained(model_args.gpt2_name_or_path, **config_kwargs)
		logger.info(f"Model config loaded from pretrained ckpt {model_args.gpt2_name_or_path}")

	config.num_labels = len(label2id)
	config.saturated = model_args.saturated
	config.onehot = model_args.onehot
	if config.onehot:
		logger.info("Using onehot embeddings.")
	config.chinese = model_args.chinese
	if config.chinese:
		logger.info("Using GPT2-Chinese.")

	# Load tokenizer
	tokenizer_kwargs = {
		"cache_dir"     : model_args.cache_dir,
		"use_fast"      : model_args.use_fast_tokenizer,
		"revision"      : model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
		}
	if model_args.tokenizer_name:
		if model_args.chinese:
			tokenizer = BertTokenizerFast.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
			logger.info("Loaded tokenizer for GPT2-Chinese.")
		else:
			tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
	elif model_args.gpt2_name_or_path:
		tokenizer = AutoTokenizer.from_pretrained(model_args.gpt2_name_or_path, **tokenizer_kwargs)
	else:
		raise ValueError(
			"You are instantiating a new tokenizer from scratch. This is not supported by this script."
			"You can do it from another script, save it, and load it from here, using --tokenizer_name."
			)
	if not model_args.chinese:
		tokenizer.pad_token = tokenizer.eos_token  # BertTokenizerFast already has pad_token, no need for GPT2-Chinese
	pre_tokenizer = WhitespaceSplit()
	tokenizer.pre_tokenizer = pre_tokenizer

	print("Vocab size of Config before tokenization: ", config.vocab_size)
	print("Vocab size of Tokenizer before tokenization: ", len(tokenizer))

	# Load GPT2 model
	if model_args.gpt2_name_or_path:
		if model_args.chinese:
			gpt2 = GPT2LMHeadModel.from_pretrained(
				model_args.gpt2_name_or_path,
				cache_dir=model_args.cache_dir,
				config=config
				)
		else:
			gpt2 = GPT2Model.from_pretrained(
				model_args.gpt2_name_or_path,
				cache_dir=model_args.cache_dir,
				config=config
				)
		logger.info(f"Model loaded from pretrained ckpt {model_args.gpt2_name_or_path}")
	elif model_args.randomized:
		gpt2 = GPT2Model(config)
		n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
		logger.info(f"Training new gpt2 from scratch - Total size={n_params / 2 ** 20:.2f}M params")
	elif model_args.mod_randomized:
		config.mod_randomized = True
		config.init_mean = model_args.init_mean
		config.init_std = model_args.init_std
		gpt2 = GPT2Model(config)
		n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
		logger.info(f"Training new gpt2 from scratch - Total size={n_params / 2 ** 20:.2f}M params")
		logger.info(f"Modified weight initialization strategy, mean: {config.init_mean}, std:{config.init_std}")
	elif model_args.agg_mod_rand:
		config.agg_mod_rand = True
		gpt2 = GPT2Model(config)
		n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
		logger.info(f"Training new gpt2 from scratch - Total size={n_params / 2 ** 20:.2f}M params")
		logger.info(f"Aggregated modified weight initialization strategy.")
	elif model_args.fine_mod_rand:
		config.fine_mod_rand = True
		gpt2 = GPT2Model(config)
		n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
		logger.info(f"Training new gpt2 from scratch - Total size={n_params / 2 ** 20:.2f}M params")
		logger.info(f"Fine modified weight initialization strategy.")

		import numpy as np
		state_dict = gpt2.state_dict()
		for name, param in state_dict.items():
			print(name)
			flattened_values: torch.Tensor = torch.flatten(param)
			flattened_values = flattened_values.detach().cpu().numpy()
			abs_values = np.absolute(flattened_values)
			mean = float(np.mean(flattened_values))
			std = float(np.std(flattened_values))
			abs_mean = float(np.mean(abs_values))
			abs_std = float(np.std(abs_values))
			print(f"Mean: {mean}, Std: {std}")
			print(f"Abs Mean: {abs_mean}, Abs Std: {abs_std}", '\n')
	elif model_args.norm_mod_rand:
		config.norm_mod_rand = True
		gpt2 = GPT2Model(config)
		n_params = sum(dict((p.data_ptr(), p.numel()) for p in gpt2.parameters()).values())
		logger.info(f"Training new gpt2 from scratch - Total size={n_params / 2 ** 20:.2f}M params")
		logger.info(f"Norm modified weight initialization strategy.")

	# Load self-defined GPT-DP model
	gpt2.resize_token_embeddings(len(tokenizer))
	config.mlp_dropout = model_args.mlp_dropout
	config.mlp_dim = model_args.mlp_dim
	config.mlp_layers = model_args.mlp_layers
	config.unary = IS_UNARY[data_args.task]
	config.use_mlp = model_args.use_mlp
	model = GPT2ForDiagnosticProbing(config, gpt2)
	record_num_of_params(model, logger)
	print("Embedding size of GPT2: ", gpt2.get_input_embeddings().weight.shape)
	print("Embedding size of MyModel: ", model.get_input_embeddings().weight.shape)

	# Preprocessing the datasets.
	# First we tokenize all the texts.
	if training_args.do_train:
		column_names = raw_datasets["train"].column_names
	else:
		column_names = raw_datasets["validation"].column_names

	def convert_span(result, pre_tokenized_str, span):
		char_start = pre_tokenized_str[span[0]][1][0]
		char_end = pre_tokenized_str[span[1]][1][1] - 1
		start = result.char_to_token(char_start)
		end = result.char_to_token(char_end)
		return [start, end]

	# Determine max_length to pad
	def pre_tokenize_function(example):
		"""
		Determine MAX_LENGTH for GPT2 model of different languages
		"""
		result = tokenizer(example['text'])
		return result

	pre_tokenized_datasets = raw_datasets.map(
		pre_tokenize_function,
		batched=False,
		num_proc=data_args.preprocessing_num_workers,
		remove_columns=column_names,
		load_from_cache_file=False,
		desc="Running tokenizer on dataset",
		)
	max_length_train = max(len(x['input_ids']) for x in pre_tokenized_datasets["train"])
	max_length_val = max(len(x['input_ids']) for x in pre_tokenized_datasets["validation"])
	max_length_test = max(len(x['input_ids']) for x in pre_tokenized_datasets["test"])
	max_length = max(max_length_train, max_length_val, max_length_test)
	print("Max length of input in Train: ", max_length_train)
	print("Max length of input in Validation: ", max_length_val)
	print("Max length of input in Test: ", max_length_test)
	print("Max length of input: ", max_length)
	del pre_tokenized_datasets

	# Dataset Tokenization
	def tokenize_function(example):
		result = tokenizer(example['text'], padding="max_length", max_length=max_length)
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

	with training_args.main_process_first(desc="dataset map tokenization"):
		print("Pad token: ", tokenizer.pad_token)
		print("Pad token ID: ", tokenizer.pad_token_id)
		tokenized_datasets = raw_datasets.map(
			tokenize_function,
			batched=False,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=False,
			desc="Running tokenizer on dataset",
			)

	if training_args.do_train:
		if "train" not in tokenized_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = tokenized_datasets["train"]
		if data_args.max_train_samples is not None:
			train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))
			total = 0
			for example in train_dataset:
				for label in example['labels']:
					if label != -1:
						total += 1
			logger.info("Total number of samples: {}".format(total))

	if training_args.do_eval:
		if "validation" not in tokenized_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = tokenized_datasets["validation"]
		if data_args.max_eval_samples is not None:
			eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

	if training_args.do_train:
		# Optimizer
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

		if model_args.do_prune:
			model.apply_dsp(model_args.num_of_heads)
			for n, p in model.named_parameters():
				if n == "gpt2.w":
					p.requires_grad = True
			optimizer_grouped_parameters.append(
				{
					"params": [p for n, p in model.named_parameters() if n == "gpt2.w"],
					"lr"    : model_args.pruning_lr,
					}
				)

		optimizer = AdamW(optimizer_grouped_parameters)
	else:
		optimizer = None

	def compute_metrics(eval_pred):
		accuracy, _ = eval_pred
		accuracy = accuracy.sum(axis=0)
		accuracy = accuracy[0] / accuracy[1]
		return {"accuracy": accuracy}

	# Modify output dir
	training_args.output_dir = os.path.join(training_args.output_dir, wandb_proj_name, serial)

	model.to(device)
	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset if training_args.do_train else None,
		eval_dataset=eval_dataset if training_args.do_eval else None,
		tokenizer=tokenizer,
		# Data collator will default to DataCollatorWithPadding, so we change it.
		data_collator=default_data_collator,
		optimizers=(optimizer, None),
		compute_metrics=compute_metrics,
		callbacks=[SaveEvalResultsCallback(), EarlyStoppingCallback(early_stopping_patience=10)],
		)

	# Training
	if training_args.do_train:
		checkpoint = None
		if training_args.resume_from_checkpoint is not None:
			checkpoint = training_args.resume_from_checkpoint
		elif last_checkpoint is not None:
			checkpoint = last_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		trainer.save_model(output_dir=training_args.output_dir)  # Saves the tokenizer too for easy upload

		metrics = train_result.metrics

		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.log_metrics("train", metrics)
	if model_args.do_prune:
		head_mask = convert_gate_to_mask(model.w, model_args.num_of_heads)
		model.apply_masks(head_mask)
		model.use_dsp = False
		logger.info("Number of heads: {}".format(head_mask.sum()))
		logger.info(f'Number of heads in each layer: {head_mask.sum(-1)}')
		if training_args.output_dir is not None:
			torch.save(head_mask, os.path.join(training_args.output_dir, "mask" + str(model_args.num_of_heads) + ".pt"))

	# Evaluation
	if training_args.do_eval:
		logger.info("*** Evaluate ***")
		logger.info(
			f'Layer weights: {torch.stack([p for n, p in model.scalar_mix.named_parameters() if "scalar" in n]).flatten()}'
			)

		metrics = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
		max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
		metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

		trainer.log_metrics("eval", metrics)

	eval_results_df.to_csv(os.path.join(training_args.output_dir, "eval_results.csv"), index=False)


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
