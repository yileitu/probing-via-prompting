import re
from math import sqrt
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel


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


def pos_neg_counts(a: np.ndarray) -> Tuple[int, int]:
	return (a > 0).sum(), (a < 0).sum()


def aggregate_gpt_module_params() -> Dict[str, np.ndarray]:
	gpt = AutoModel.from_pretrained("gpt2")
	gpt_params = gpt.named_parameters()
	gpt_agg_params: Dict[str, np.ndarray] = {}
	for name, values in gpt_params:
		values = torch.flatten(values).detach().cpu().numpy()
		matched = re.match(MATCH_RULE, name)
		if matched:
			agg_name = matched.group(0)
			if agg_name not in gpt_agg_params:
				gpt_agg_params[agg_name] = values
			else:
				gpt_agg_params[agg_name] = np.append(gpt_agg_params[agg_name], values)
		else:
			gpt_agg_params[name] = values

	return gpt_agg_params


if __name__ == "__main__":
	GPT_DF_COLS: List[str] = ["module_name", "total_num", "pos_cnt", "neg_cnt", "diff", "diff_ratio", "mean", "abs_mean", "std",
	                          "abs_std"]
	MATCH_RULE: str = r"h\.\d{1,2}"  # Match prefix "h.1" format

	gpt = AutoModel.from_pretrained("gpt2")
	gpt_config = AutoConfig.from_pretrained("gpt2")
	gpt_params = gpt.named_parameters()
	gpt_module_stat_df: pd.DataFrame = pd.DataFrame(columns=GPT_DF_COLS)

	# all_param_values = torch.tensor([])
	# for name, values in gpt_params:
	# 	pprint(name)
	# 	pprint(type(name))
	# 	flattened_values = torch.flatten(values)
	# 	all_param_values = torch.cat((all_param_values, flattened_values))
	# print(all_param_values.size())
	# mean = torch.mean(all_param_values).item()
	# std = torch.std(all_param_values).item()
	# print(mean, std)

	# for name, values in gpt_params:
	# 	flattened_values: torch.Tensor = torch.flatten(values)
	# 	flattened_values: np.ndarray = flattened_values.detach().cpu().numpy()
	#
	# 	pos_cnt, neg_cnt = pos_neg_counts(flattened_values)
	# 	diff = abs(pos_cnt - neg_cnt)
	# 	diff_ratio = diff / (pos_cnt + neg_cnt)
	#
	# 	mean = float(np.mean(flattened_values))
	# 	std = float(np.std(flattened_values))
	#
	# 	abs_values = np.absolute(flattened_values)
	# 	abs_mean = float(np.mean(abs_values))
	# 	abs_std = float(np.std(abs_values))
	#
	# 	datapoint = {
	# 		"module_name": name,
	# 		"pos_cnt"    : pos_cnt,
	# 		"neg_cnt"    : neg_cnt,
	# 		"diff"       : diff,
	# 		"diff_ratio" : diff_ratio,
	# 		"mean"       : mean,
	# 		"abs_mean"   : abs_mean,
	# 		"std"        : std,
	# 		"abs_std"    : abs_std,
	# 		}
	# 	datapoint_df = pd.DataFrame([datapoint])
	# 	gpt_module_stat_df = pd.concat([gpt_module_stat_df, datapoint_df])
	#
	# gpt_module_stat_df.to_csv("gpt_module_stat.csv")

	# gpt_module_agg_stat_df: pd.DataFrame = pd.DataFrame(columns=GPT_DF_COLS)
	# gpt_agg_params = aggregate_gpt_module_params()
	# for name, values in gpt_agg_params.items():
	# 	total_num = len(values)
	# 	pos_cnt, neg_cnt = pos_neg_counts(values)
	# 	diff = abs(pos_cnt - neg_cnt)
	# 	diff_ratio = diff / (pos_cnt + neg_cnt)
	#
	# 	mean = float(np.mean(values))
	# 	std = float(np.std(values))
	#
	# 	abs_values = np.absolute(values)
	# 	abs_mean = float(np.mean(abs_values))
	# 	abs_std = float(np.std(abs_values))
	#
	# 	datapoint = {
	# 		"total_num"  : total_num,
	# 		"module_name": name,
	# 		"pos_cnt"    : pos_cnt,
	# 		"neg_cnt"    : neg_cnt,
	# 		"diff"       : diff,
	# 		"diff_ratio" : diff_ratio,
	# 		"mean"       : mean,
	# 		"abs_mean"   : abs_mean,
	# 		"std"        : std,
	# 		"abs_std"    : abs_std,
	# 		}
	# 	datapoint_df = pd.DataFrame([datapoint])
	# 	gpt_module_agg_stat_df = pd.concat([gpt_module_agg_stat_df, datapoint_df])
	#
	# gpt_module_agg_stat_df.to_csv("gpt_module_agg_stat.csv")

	# for name, module in gpt.named_modules():
	# 	print(name)
	# 	print(module, '\n', type(module), '\n')

