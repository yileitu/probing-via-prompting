from copy import deepcopy
from typing import Optional

import torch
from allennlp.modules import scalar_mix
from allennlp.modules.span_extractors import SelfAttentiveSpanExtractor
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.file_utils import ModelOutput

import wandb
from utils import STEFunction


class DiagnosticProbingOutputs(ModelOutput):
	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None


class GPT2ForDiagnosticProbing(GPT2PreTrainedModel):
	def __init__(self, config, gpt2):
		super().__init__(config)
		if config.chinese:
			self.transformer = gpt2.transformer  # To get of rid of LMHead of GPT2LMHeadModel (GPT2-Chinese)
		else:
			self.transformer = gpt2

		if config.onehot is False:
			for param in self.transformer.parameters():
				param.requires_grad = False
		else:
			for param in self.transformer.parameters():
				param.requires_grad = True
			print("Onehot is True. All parameters are trainable.")

		# Model parallel
		self.model_parallel = False
		self.device_map = None

		self.unary = config.unary
		self.num_labels = config.num_labels
		self.mlp_dropout = config.mlp_dropout
		self.mlp_dim = config.mlp_dim
		self.mlp_layers: int = config.mlp_layers
		self.use_mlp = config.use_mlp
		self.n_embd = config.n_embd

		self.onehot: bool = config.onehot

		self.scalar_mix = scalar_mix.ScalarMix(config.n_layer)
		# self.onehot_scalar_mix = scalar_mix.ScalarMix(1)

		if self.onehot is False:
			self.proj1 = nn.Conv1d(
				config.n_embd,
				config.mlp_dim,
				kernel_size=1,
				stride=1,
				padding=0,
				dilation=1,
				groups=1,
				bias=True,
				)
		else:
			self.proj1 = nn.Conv1d(
				config.vocab_size,
				config.mlp_dim,
				kernel_size=1,
				stride=1,
				padding=0,
				dilation=1,
				groups=1,
				bias=True,
				)
		self.span_extractor1 = SelfAttentiveSpanExtractor(config.mlp_dim)
		self.d_inp = self.span_extractor1.get_output_dim()
		if not self.unary:
			if self.onehot is False:
				self.proj2 = nn.Conv1d(
					config.n_embd,
					config.mlp_dim,
					kernel_size=1,
					stride=1,
					padding=0,
					dilation=1,
					groups=1,
					bias=True,
					)
			else:
				self.proj2 = nn.Conv1d(
					config.vocab_size,
					config.mlp_dim,
					kernel_size=1,
					stride=1,
					padding=0,
					dilation=1,
					groups=1,
					bias=True,
					)
			self.span_extractor2 = SelfAttentiveSpanExtractor(config.mlp_dim)
			self.d_inp += self.span_extractor2.get_output_dim()

		wandb.run.summary["input_layer_dim"] = self.d_inp

		if not self.use_mlp:
			lin_module_list = []
			if self.mlp_layers == 1:
				self.classifier = nn.Sequential(
					nn.Linear(self.d_inp, self.mlp_dim),
					nn.Linear(self.mlp_dim, self.num_labels)
					)
			elif self.mlp_layers >= 2:
				lin_module_list.append(nn.Linear(self.d_inp, self.mlp_dim))
				for _ in range(self.mlp_layers - 1):
					lin_module_list.append(nn.Linear(self.mlp_dim, self.mlp_dim))
				lin_module_list.append(nn.Linear(self.mlp_dim, self.num_labels))
				self.classifier = nn.Sequential(*lin_module_list)
		else:
			input_layer_list = [
				nn.Linear(self.d_inp, self.mlp_dim),
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

		self.w = nn.Parameter(torch.empty([config.num_hidden_layers, config.num_attention_heads]))
		nn.init.xavier_uniform(self.w)
		self.num_of_heads = None
		self.use_dsp = False

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
			output_hidden_states=None,
			return_dict=None,
			span1s=None,
			span2s=None,
			):
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict

		if self.use_dsp:
			head_mask = STEFunction.apply(self.w.view(-1), self.num_of_heads).view_as(self.w)
			self.apply_masks(head_mask)

		if self.onehot is False:
			transformer_outputs = self.transformer(
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
				return_dict=True,
				)
			if not self.use_mlp:
				contextual_embeddings = transformer_outputs[0]
			else:
				all_hidden_states = transformer_outputs.hidden_states[1:]
				contextual_embeddings = self.scalar_mix(all_hidden_states)
		else:
			contextual_embeddings = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size).half()

		span_mask = span1s[:, :, 0] != -1
		se_proj1 = self.proj1(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
		span1_emb = self.span_extractor1(se_proj1, span1s, span_indices_mask=span_mask.long())
		if not self.unary:
			se_proj2 = self.proj2(contextual_embeddings.transpose(1, 2)).transpose(2, 1).contiguous()
			span2_emb = self.span_extractor2(se_proj2, span2s, span_indices_mask=span_mask.long())
			span_emb = torch.cat([span1_emb, span2_emb], dim=2)
		else:
			span_emb = span1_emb

		logits = self.classifier(span_emb)
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits[span_mask], labels[span_mask])

		corrections = logits[span_mask].argmax(-1) == labels[span_mask]
		correct_counts = corrections.sum()
		total_counts = len(corrections)
		accuracy = torch.tensor([[correct_counts, total_counts]], device=corrections.device)

		if not return_dict:
			output = (accuracy,)
			return ((loss,) + output) if loss is not None else output

		return DiagnosticProbingOutputs(
			loss=loss,
			logits=accuracy,
			)

	def apply_masks(self, head_mask):
		head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
		self.transformer.apply_masks(head_mask)

	def get_masks(self):
		return torch.stack(self.transformer.get_masks())

	def apply_dsp(self, num_of_heads):
		self.num_of_heads = num_of_heads
		self.use_dsp = True
