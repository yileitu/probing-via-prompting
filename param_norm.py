from math import sqrt
from pprint import pprint
from typing import Dict, Iterable, Tuple

import torch
from transformers import AutoModel

import numpy as np


def _fan_in(shape) -> int:
	# This is from some TensorFlow code or something.
	return float(shape[-2]) if len(shape) > 1 else float(shape[-1])


def get_param_norm(params: Iterable[np.ndarray], normalize: bool = False, min: bool = False) -> float:
	# There are weird scalars in here, which we filter out.
	values = [v for v in params if len(v.shape) > 0]
	if min:
		# Take the linear transformation in the network with the least norm.
		values = [v / np.sqrt(v.size) for v in values if len(v.shape) == 2]
		norms = [np.linalg.norm(v) for v in values]
		return np.min(norms)
	else:
		# This is the 2-norm.
		if normalize:
			values = [value / sqrt(_fan_in(value.shape)) for value in values]
		flat = np.concatenate([value.flatten() for value in values])
		norm = np.linalg.norm(flat)
		return norm


def get_param(params: Iterable[np.ndarray]) -> np.ndarray:
	values = [v for v in params if len(v.shape) > 0]
	return np.concatenate([value.flatten() for value in values])


def filter_by_layer(param_names, layer_num: int):
	expr = f"encoder/block_{layer_num:03d}"
	return [p for p in param_names if p.startswith(expr)]


if __name__ == "__main__":
	gpt = AutoModel.from_pretrained("gpt2")
	gpt_params = gpt.named_parameters()
	# pprint(gpt_params)
	gpt_module_mean_std: Dict[str, Tuple[float, float]] = dict()

	for name, values in gpt_params:
		flattened_values = torch.flatten(values)
		mean = torch.mean(flattened_values).item()
		std = torch.std(flattened_values).item()
		gpt_module_mean_std[name] = (mean, std)

	pprint(gpt_module_mean_std)

	# all_param_values = torch.tensor([])
	# for name, values in gpt_params:
	# 	pprint(name)
	# 	pprint(type(name))
	# 	flattened_values = torch.flatten(values)
	# 	all_param_values = torch.cat((all_param_values, flattened_values))
	#
	# print(all_param_values.size())
	#
	# mean = torch.mean(all_param_values).item()
	# std = torch.std(all_param_values).item()
	#
	# print(mean, std)
