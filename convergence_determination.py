# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

DELTA_EPOCH: int = 10
CONVERGENCE_THRESHOLD: float = 0.001
DIM: int = 256
LAYER: int = 8
LEN: int = 50
LR: str = "0.001"
# LR: str = "5e-05"


def find_convergence_point(accuracies: List[float], window_size: int = 16, threshold=1e-4):
	avg_diffs = []

	# Calculate the differences between consecutive accuracies and store them in avg_diffs
	for i in range(1, len(accuracies)):
		avg_diffs.append(abs(accuracies[i] - accuracies[i - 1]))

	for i in range(len(avg_diffs) - window_size + 1):
		avg_diff = sum(avg_diffs[i:i + window_size]) / window_size
		if avg_diff < threshold:
			return i + window_size  # Return the index of the last epoch in the window
	return -1  # If convergence is not found


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


# acc_df = pd.read_csv(
# 	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-DPMLP-Dim{DIM}-Layer{LAYER}/Epoch256-LR{LR}-Randomized-Test/eval_results.csv'
# 	)
acc_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-PP-flat-Len{LEN}/Epoch256-LR{LR}-Randomized-Test/eval_results.csv'
	)
acc = acc_df['eval_accuracy'].values.tolist()[0:-1]

# convergence_epoch = find_convergence_point(acc, window_size=DELTA_EPOCH, threshold=CONVERGENCE_THRESHOLD)
# if convergence_epoch != -1:
# 	print(f"Convergence occurs at epoch {convergence_epoch}, with accuracy {acc[convergence_epoch]:.4f}")
# else:
# 	print("No convergence found")

# convergence = find_convergence_epoch(
# 	accuracy_data=acc, threshold=CONVERGENCE_THRESHOLD, n_consecutive_epochs=DELTA_EPOCH
# 	)
# if convergence:
# 	epoch, accuracy = convergence
# 	print(f"Convergence found at epoch {epoch}, with accuracy {accuracy:.4f}.")
# else:
# 	print("Convergence not found.")

# # Set the window size for computing the standard deviation
# window_size = 10
# # Compute the standard deviation of accuracy values for each window
# std_devs = [np.std(acc[i:i + window_size]) for i in range(len(acc) - window_size)]
# # Find the first epoch at which the standard deviation falls below a threshold
# threshold = 0.003
# converged_epoch = next((i for i, std_dev in enumerate(std_devs) if std_dev < threshold), len(acc))
# converged_acc = acc[converged_epoch]
# print(f"Convergence found at epoch {converged_epoch}, with accuracy {converged_acc:.4f}.")

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Define an exponential function to fit to the accuracy values
# def exp_func(x, a, b, c):
# 	return a * np.exp(-b * x) + c

# Define a function to fit to the accuracy values
def fit_func(x, a, b, c):
	# return c - a * x ** (-b)
	return a * np.exp(-b * x) + c


# Fit the function to the accuracy values
popt, _ = curve_fit(fit_func, range(len(acc)), acc)
print(*popt)

# Compute the fitted curve using the optimal parameters
fitted_curve = fit_func(range(len(acc)), *popt)


# Compute the derivative of the fitted function
def deriv_func(x, a, b, c):
	# return a * b * x ** (-(b + 1))
	return -a * b * np.exp(-b * x)


# Compute the derivative values
deriv_values = deriv_func(range(1, len(acc) + 1), *popt)

# Find the first epoch at which the derivative falls below the threshold
threshold = 0.0001
converged_epoch = next((i for i, deriv in enumerate(deriv_values) if abs(deriv) < threshold), len(acc))
converged_acc = acc[converged_epoch]
print(f"Convergence found at epoch {converged_epoch}, with accuracy {converged_acc:.4f}.")

# Plot the fitted curve and the original data on the same plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(len(acc)), acc, marker='o', markersize=4, linestyle='-', label='Original data')
ax.plot(range(len(acc)), fitted_curve, color='r', linewidth=2, linestyle='-', label='Fitted curve')
ax.legend(fontsize=12)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title('Accuracy vs Epoch', fontsize=16)
ax.grid(True)
plt.show()
