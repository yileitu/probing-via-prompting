# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import pandas as pd

DELTA_EPOCH: int = 10
CONVERGENCE_THRESHOLD: float = 0.01
DIM: int = 512
LAYER: int = 8
LR: str = "0.0001"


# def find_convergence_point(accuracies: List[float], window_size: int = 16, threshold=1e-4):
# 	avg_diffs = []
#
# 	# Calculate the differences between consecutive accuracies and store them in avg_diffs
# 	for i in range(1, len(accuracies)):
# 		avg_diffs.append(abs(accuracies[i] - accuracies[i - 1]))
#
# 	for i in range(len(avg_diffs) - window_size + 1):
# 		avg_diff = sum(avg_diffs[i:i + window_size]) / window_size
# 		if avg_diff < threshold:
# 			return i + window_size  # Return the index of the last epoch in the window
# 	return -1  # If convergence is not found

def find_convergence_epoch(accuracy_data: List[float], threshold: float = 1e-3, n_consecutive_epochs: int = 10) \
		-> Optional[Tuple[int, float]]:
	# get the largest accuracy value
	max_acc = max(accuracy_data)

	# loop through the accuracy data from the beginning
	for i in range(len(accuracy_data) - n_consecutive_epochs + 1):
		# get the subset of accuracy data for the current set of consecutive epochs
		subset = accuracy_data[i:i + n_consecutive_epochs]

		# check if all the values in the subset are within the threshold of the max accuracy
		if all(abs(max_acc - x) < threshold for x in subset):
			# if they are, we've found convergence!
			convergence_epoch = i + n_consecutive_epochs
			convergence_accuracy = subset[-1]
			return convergence_epoch, convergence_accuracy

	# if we didn't find convergence in any subset of consecutive epochs, return None
	return None


acc_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-DPMLP-Dim{DIM}-Layer{LAYER}/Epoch256-LR{LR}-Randomized-Test/eval_results.csv'
	)
acc = acc_df['eval_accuracy'].values.tolist()[0:-1]

# convergence_epoch = find_convergence_point(acc, window_size=DELTA_EPOCH, threshold=CONVERGENCE_THRESHOLD)
# if convergence_epoch != -1:
# 	print(f"Convergence occurs at epoch {convergence_epoch}, with accuracy {acc[convergence_epoch]:.4f}")
# else:
# 	print("No convergence found")

convergence = find_convergence_epoch(
	accuracy_data=acc, threshold=CONVERGENCE_THRESHOLD, n_consecutive_epochs=DELTA_EPOCH
	)
if convergence:
	epoch, accuracy = convergence
	print(f"Convergence found at epoch {epoch}, with accuracy {accuracy}.")
else:
	print("Convergence not found.")
