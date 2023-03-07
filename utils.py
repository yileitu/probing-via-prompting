from typing import Any, Dict

import torch

LABEL_DICT = {}
LABEL_DICT['ner'] = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE',
                     'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT',
                     'QUANTITY', 'TIME', 'WORK_OF_ART']
LABEL_DICT['pos'] = ['$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX',
                     'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
                     'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
                     'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                     'WDT', 'WP', 'WP$', 'WRB', '``']
LABEL_DICT['const'] = ['ADJP', 'ADVP', 'CONJP', 'EMBED', 'FRAG', 'INTJ', 'LST',
                       'META', 'NAC', 'NML', 'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'S', 'SBAR',
                       'SBARQ', 'SINV', 'SQ', 'TOP', 'UCP', 'VP', 'WHADJP', 'WHADVP', 'WHNP', 'WHPP',
                       'X']
LABEL_DICT['coref'] = ['False', 'True']
LABEL_DICT['srl'] = ['ARG0', 'ARG1', 'ARG2', 'ARG3', 'ARG4', 'ARG5', 'ARGA',
                     'ARGM-ADJ', 'ARGM-ADV', 'ARGM-CAU', 'ARGM-COM', 'ARGM-DIR', 'ARGM-DIS', 'ARGM-DSP',
                     'ARGM-EXT', 'ARGM-GOL', 'ARGM-LOC', 'ARGM-LVB', 'ARGM-MNR', 'ARGM-MOD', 'ARGM-NEG',
                     'ARGM-PNC', 'ARGM-PRD', 'ARGM-PRP', 'ARGM-PRR', 'ARGM-PRX', 'ARGM-REC', 'ARGM-TMP',
                     'C-ARG0', 'C-ARG1', 'C-ARG2', 'C-ARG3', 'C-ARG4', 'C-ARGM-ADJ', 'C-ARGM-ADV',
                     'C-ARGM-CAU', 'C-ARGM-COM', 'C-ARGM-DIR', 'C-ARGM-DIS', 'C-ARGM-DSP', 'C-ARGM-EXT',
                     'C-ARGM-LOC', 'C-ARGM-MNR', 'C-ARGM-MOD', 'C-ARGM-NEG', 'C-ARGM-PRP', 'C-ARGM-TMP',
                     'R-ARG0', 'R-ARG1', 'R-ARG2', 'R-ARG3', 'R-ARG4', 'R-ARG5', 'R-ARGM-ADV', 'R-ARGM-CAU',
                     'R-ARGM-COM', 'R-ARGM-DIR', 'R-ARGM-EXT', 'R-ARGM-GOL', 'R-ARGM-LOC', 'R-ARGM-MNR',
                     'R-ARGM-MOD', 'R-ARGM-PNC', 'R-ARGM-PRD', 'R-ARGM-PRP', 'R-ARGM-TMP']
for task in LABEL_DICT:
	LABEL_DICT[task] = {label: "label" + str(i) for i, label in enumerate(LABEL_DICT[task])}


def convert_gate_to_mask(gates, num_of_heads=None):
	if num_of_heads is not None:
		head_mask = torch.zeros_like(gates)
		current_heads_to_keep = gates.view(-1).sort(descending=True)[1]
		current_heads_to_keep = current_heads_to_keep[:num_of_heads]
		head_mask = head_mask.view(-1)
		head_mask[current_heads_to_keep] = 1.0
		head_mask = head_mask.view_as(gates)
	else:
		head_mask = (gates > 0.5).float()
	return head_mask


class STEFunction(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input, k):
		threshold = input.sort(descending=True)[0][k]
		return (input > threshold).float()

	@staticmethod
	def backward(ctx, grad_output):
		return grad_output, None


def transform_dict(config_dict: Dict, expand: bool = True):
	"""
	General function to transform any dictionary into wandb config acceptable format
	(This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
	The expand argument is used to expand iterables into dictionaries so that these configs can be used when compare across runs
	"""
	ret: Dict[str, Any] = {}
	for k, v in config_dict.items():
		if v is None or isinstance(v, (int, float, str)):
			ret[k] = v
		elif isinstance(v, (list, tuple, set)):
			# Need to check if item in iterable is YAML-friendly
			t = transform_dict(dict(enumerate(v)), expand)
			# Transform back to iterable if expand is False
			ret[k] = t if expand else [t[i] for i in range(len(v))]
		elif isinstance(v, dict):
			ret[k] = transform_dict(v, expand)
		else:
			# Transform to YAML-friendly (str) format
			# Need to handle both Classes, Callables, Object Instances
			# Custom Classes might not have great __repr__ so __name__ might be better in these cases
			vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
			ret[k] = f"{v.__module__}:{vname}"
	return ret


def hardmax2(t):
	idx = t.argmax(dim=-1).view(-1)
	_t = 1
	for i in t.shape[:-1]:
		_t *= i
	_range = torch.arange(_t, device=t.device)
	step = t.shape[-1]
	_range *= step
	idx += _range
	res = torch.zeros_like(t).view(-1)
	res[idx] = 1.
	return res.view(t.shape)


def hardmax(X):
	M, _ = torch.max(X, dim=-1, keepdim=True)
	A = (M == X).float()
	A /= torch.sum(A, dim=-1, keepdim=True)

	return A


# To test hardmax functions
# pre_x = [[-10, 2, 2, 2], [-100, 1, 0, 1]]
# X = torch.Tensor(pre_x)
# print(hardmax(X))
#
# for num_dims in range(1, 6):
# 	pre_x = [[-10, 2, 2, 2], [-100, 1, 0, 1]]
# 	for _ in range(num_dims - 1):
# 		pre_x = [pre_x]
# 		X = torch.Tensor(pre_x)
# 		print(X)
# 		print(hardmax2(X), '\n')


def bimodal_normal(x: torch.Tensor, mu: float, sigma: float) -> None:
	"""
	Inits the weights (in-place) with the bimodal normal distribution (symmetric).

	:param x: input tensor
	:param mu: mean of the normal distribution
	:param sigma: standard deviation of the normal distribution
	"""
	x.normal_(mean=mu, std=sigma)
	# size = x.size()
	# mask = torch.randint(0, 2, size=size) * 2 - 1  # Randomly flip half the values to their opposite sign
	# x *= mask
