#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (AdamW, AutoConfig, AutoTokenizer, CONFIG_MAPPING, HfArgumentParser,
                          MODEL_FOR_CAUSAL_LM_MAPPING, Trainer, TrainingArguments, default_data_collator, set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

import wandb
from modeling_gated_gpt2 import GPT2Model
from modeling_gpt2_dp import GPT2ForDiagnosticProbing
from utils import LABEL_DICT, convert_gate_to_mask, transform_dict

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

MAX_LENGTH = {'pos': 350, 'const': 350, 'ner': 350, 'coref': 280, 'srl': 350}
MAX_TARGET = {'pos': 275, 'const': 175, 'ner': 71, 'coref': 300, 'srl': 11}
IS_UNARY = {'pos': True, 'const': True, 'ner': True, 'coref': False, 'srl': False}


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
	"""

	gpt2_name_or_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "The model checkpoint for weights initialization."
			        "Don't set if you want to train a model from scratch."
			},
		)
	model_path: Optional[str] = field(
		default=None,
		metadata={
			"help": "Path to trained model."
			        "Don't set if you want to train a model from scratch."
			},
		)
	model_type: Optional[str] = field(
		default=None,
		metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
		)
	config_overrides: Optional[str] = field(
		default=None,
		metadata={
			"help": "Override some existing default config settings when a model is trained from scratch. Example: "
			        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
			},
		)
	config_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
		)
	tokenizer_name: Optional[str] = field(
		default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
		)
	cache_dir: Optional[str] = field(
		default='cache/',
		metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
		)
	use_fast_tokenizer: bool = field(
		default=True,
		metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
		)
	model_revision: str = field(
		default="main",
		metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
		)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
			        "with private models)."
			},
		)
	use_mlp: bool = field(
		default=True,
		metadata={
			"help": "use mlp or linear regression"
			},
		)
	mlp_dropout: Optional[float] = field(
		default=0.2,
		metadata={"help": "Dropout in MLP model."},
		)
	mlp_dim: Optional[int] = field(
		default=512,
		metadata={"help": "Dimension of hidden states of MLP model."},
		)
	mlp_layers: Optional[int] = field(
		default=1,
		metadata={"help": "The number of layers of MLP model."},
		)
	num_of_heads: Optional[int] = field(
		default=96,
		metadata={"help": "Number of heads left unpruned."},
		)
	pruning_lr: Optional[float] = field(
		default=0.1,
		metadata={"help": "Learning rate for head importance variables."},
		)
	do_prune: Optional[bool] = field(
		default=False,
		metadata={"help": "Whether heads are pruned."},
		)
	randomized: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights. "
			        "By default (randomized=False), load the whole pretrained model."
			},
		)
	dev: bool = field(
		default=False,
		metadata={
			"help": "If true, use development dataset to do evaluation. Otherwise use test dataset."
			},
		)
	mod_randomized: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights. "
			        "Artificially specify how to initialize the weights, e.g., init_mean, init_std, etc."
			},
		)
	agg_mod_rand: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights."
			        "Initialize the weights head by head with predetermined parameters, e.g., abs_mean, abs_std, etc."
			},
		)
	fine_mod_rand: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights."
			        "Initialize the weights module by module with predetermined parameters, e.g., abs_mean, abs_std, etc."
			},
		)
	norm_mod_rand: bool = field(
		default=False,
		metadata={
			"help": "If true, load the architecture of the model only, without pretrained weights."
			        "Initialize the weights module by module with specified norm."
			},
		)
	init_mean: float = field(
		default=0.0,
		metadata={
			"help": "Randomized model weight initialization mean"
			},
		)
	init_std: float = field(
		default=0.02,
		metadata={
			"help": "Randomized model weight initialization std"
			},
		)
	verbose: int = field(
		default=0,
		metadata={
			"help": "How to group wandb experiments."
			},
		)
	saturated: bool = field(
		default=False,
		metadata={
			"help": "Saturated attention mode."
			},
		)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	data_dir: Optional[str] = field(
		default=None, metadata={"help": "Where data is stored"}
		)
	task: Optional[str] = field(
		default='ner',
		metadata={"help": "Tasks, one or more of {pos, const, coref, ner, srl}."},
		)
	max_train_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of training examples to this "
			        "value if set."
			},
		)
	max_eval_samples: Optional[int] = field(
		default=None,
		metadata={
			"help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
			        "value if set."
			},
		)
	overwrite_cache: bool = field(
		default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
		)
	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
		)


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
	# Pretrained or Randomized
	if model_args.randomized or model_args.mod_randomized or model_args.agg_mod_rand or model_args.fine_mod_rand \
			or model_args.norm_mod_rand:
		model_args.gpt2_name_or_path = None
		model_args.config_name = "gpt2"
		model_args.tokenizer_name = "gpt2"

	# Determine the default experiment serial
	serial = ""
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

	if model_args.mod_randomized:
		group_name = f"Mean{model_args.init_mean}-Std{model_args.init_std}"
	else:
		group_name = f"Epoch{int(training_args.num_train_epochs)}-LR{training_args.learning_rate}-WD{training_args.weight_decay}"

	# WanDB setup
	if model_args.use_mlp:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}"
	else:
		wandb_proj_name = "Probe-" + data_args.task + "-DP-LR"
	if model_args.mod_randomized:
		wandb_proj_name += "-ModRand"

	# CONCERN: 写得不优美，先用verbose代替处理如何控制wandb分组
	if model_args.verbose == 1 and model_args.mod_randomized:
		wandb_proj_name = f"Probe-{data_args.task}-DP-MLP-ModRand-Mean{model_args.init_mean}-Std{model_args.init_std}"
		group_name = f"Dim{model_args.mlp_dim}-Layer{model_args.mlp_layers}-Epoch{int(training_args.num_train_epochs)}"
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

	os.environ["WANDB_PROJECT"] = wandb_proj_name
	wandb.init(
		project=wandb_proj_name,
		name=serial,
		group=group_name,
		)
	training_args.report_to = ["wandb"]
	training_args.logging_steps = 50
	training_args.run_name = serial

	wandb.log(transform_dict(asdict(model_args)))
	wandb.log(transform_dict(asdict(data_args)))

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		handlers=[logging.StreamHandler(sys.stdout)],
		)

	log_level = training_args.get_process_log_level()
	logger.setLevel(log_level)
	datasets.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.set_verbosity(log_level)
	transformers.utils.logging.enable_default_handler()
	transformers.utils.logging.enable_explicit_format()

	# Log on each process the small summary:
	logger.warning(
		f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
		+ f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
		)
	logger.info(f"Training/evaluation parameters {training_args}")

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

	# Set seed before initializing model.
	set_seed(training_args.seed)

	# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
	# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
	# (the dataset will be downloaded automatically from the datasets Hub).
	#
	# For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
	# 'text' is found. You can easily tweak this behavior (see below).
	#
	# In distributed training, the load_dataset function guarantee that only one local process can concurrently
	# download the dataset.

	data_files = {}
	dataset_args = {}
	logger.info("Loading data for {}".format(data_args.task))
	if training_args.do_train:
		data_files["train"] = os.path.join(data_args.data_dir, data_args.task, 'train.json')

	if model_args.dev:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'development.json')
	else:
		data_files["validation"] = os.path.join(data_args.data_dir, data_args.task, 'test.json')

	raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)
	if "_control" in data_args.task:
		data_args.task = data_args.task.replace("_control", "")
	label2id = {label: i for i, label in enumerate(LABEL_DICT[data_args.task])}

	# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
	# https://huggingface.co/docs/datasets/loading_datasets.html.

	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.

	config_kwargs = {
		"cache_dir"     : model_args.cache_dir,
		"revision"      : model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
		}
	if model_args.config_name:
		config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
	elif model_args.gpt2_name_or_path:
		config = AutoConfig.from_pretrained(model_args.gpt2_name_or_path, **config_kwargs)
	else:
		config = CONFIG_MAPPING[model_args.model_type]()
		logger.warning("You are instantiating a new config instance from scratch.")
		if model_args.config_overrides is not None:
			logger.info(f"Overriding config: {model_args.config_overrides}")
			config.update_from_string(model_args.config_overrides)

	tokenizer_kwargs = {
		"cache_dir"     : model_args.cache_dir,
		"use_fast"      : model_args.use_fast_tokenizer,
		"revision"      : model_args.model_revision,
		"use_auth_token": True if model_args.use_auth_token else None,
		}
	if model_args.tokenizer_name:
		tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
	elif model_args.gpt2_name_or_path:
		tokenizer = AutoTokenizer.from_pretrained(model_args.gpt2_name_or_path, **tokenizer_kwargs)
	else:
		raise ValueError(
			"You are instantiating a new tokenizer from scratch. This is not supported by this script."
			"You can do it from another script, save it, and load it from here, using --tokenizer_name."
			)

	config.num_labels = len(label2id)
	config.saturated = model_args.saturated

	if model_args.gpt2_name_or_path:
		gpt2 = GPT2Model.from_pretrained(
			model_args.gpt2_name_or_path,
			cache_dir=model_args.cache_dir,
			config=config
			)
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

		# import numpy as np
		# state_dict = gpt2.state_dict()
		# for name, param in state_dict.items():
		# 	print(name)
		# 	norm = torch.norm(param)
		# 	print(f"Norm: {norm}")

	gpt2.resize_token_embeddings(len(tokenizer))

	if model_args.model_path:
		# TODO: Pretrained model 如何更改GPT2的mlp
		config = AutoConfig.from_pretrained(model_args.model_path, cache_dir=model_args.cache_dir)
		model = GPT2ForDiagnosticProbing.from_pretrained(model_args.model_path, config=config, gpt2=gpt2)
	else:
		config.mlp_dropout = model_args.mlp_dropout
		config.mlp_dim = model_args.mlp_dim
		config.mlp_layers = model_args.mlp_layers
		config.unary = IS_UNARY[data_args.task]
		config.use_mlp = model_args.use_mlp
		model = GPT2ForDiagnosticProbing(config, gpt2)

	# Preprocessing the datasets.
	# First we tokenize all the texts.
	if training_args.do_train:
		column_names = raw_datasets["train"].column_names
	else:
		column_names = raw_datasets["validation"].column_names

	# since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
	tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

	tokenizer.pad_token = tokenizer.eos_token
	pre_tokenizer = WhitespaceSplit()
	tokenizer.pre_tokenizer = pre_tokenizer

	# Record num of params
	num_trainable_params = model.num_parameters(only_trainable=True)
	num_total_params = model.num_parameters()
	logger.info(f"Number of parameters to train (without adapters): {num_trainable_params}")
	logger.info(f"Total number of parameters (without adapters): {num_total_params}")
	wandb.run.summary["num_trainable_params"] = num_trainable_params
	wandb.run.summary["num_total_params"] = num_total_params

	def convert_span(result, pre_tokenized_str, span):
		char_start = pre_tokenized_str[span[0]][1][0]
		char_end = pre_tokenized_str[span[1]][1][1] - 1
		start = result.char_to_token(char_start)
		end = result.char_to_token(char_end)
		return [start, end]

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

	with training_args.main_process_first(desc="dataset map tokenization"):
		tokenized_datasets = raw_datasets.map(
			tokenize_function,
			batched=False,
			num_proc=data_args.preprocessing_num_workers,
			remove_columns=column_names,
			load_from_cache_file=not data_args.overwrite_cache,
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
		)

	# Training
	if training_args.do_train:
		checkpoint = None
		if training_args.resume_from_checkpoint is not None:
			checkpoint = training_args.resume_from_checkpoint
		elif last_checkpoint is not None:
			checkpoint = last_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)
		# trainer.save_model()  # Saves the tokenizer too for easy upload

		metrics = train_result.metrics

		max_train_samples = (
			data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
		)
		metrics["train_samples"] = min(max_train_samples, len(train_dataset))

		trainer.log_metrics("train", metrics)
	# trainer.save_metrics("train", metrics)
	# trainer.save_state()

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

		metrics = trainer.evaluate()

		max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
		metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

		trainer.log_metrics("eval", metrics)


# trainer.save_metrics("eval", metrics)


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main()
