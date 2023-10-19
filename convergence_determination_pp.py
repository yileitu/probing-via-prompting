# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

DELTA_EPOCH: int = 10
CONVERGENCE_THRESHOLD: float = 0.001
LEN: int = 200
# LR: str = "0.0001"
LR: str = "5e-06"

if LEN == 50:
	warmup = 4
elif LEN == 100:
	warmup = 20
elif LEN == 200:
	warmup = 52

dev_acc_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-PP-Len{LEN}/Epoch256-LR{LR}-Randomized-Dev/eval_results.csv'
	)
dev_acc = dev_acc_df['eval_accuracy'].values.tolist()[0:-1]
dev_acc = list(map(lambda x: x * 100, dev_acc))  # convert to percentage


# Define a function to fit to the accuracy values
def fit_func(x, a, b, c):
	# return c - a * x ** (-b)
	return a * np.exp(-b * x) + c


# Compute the derivative of the fitted function
def deriv_func(x, a, b, c):
	# return a * b * x ** (-(b + 1))
	return -a * b * np.exp(-b * x)


y_dev = dev_acc[warmup:]
x_dev = np.arange(1, len(y_dev) + 1)

# Fit the function to the accuracy values
initial_guess = [1, 0.01, 80]
popt_dev, _ = curve_fit(fit_func, x_dev, y_dev, initial_guess)

# Compute the fitted curve using the optimal parameters
fitted_dev = fit_func(x_dev, *popt_dev)

# Compute the derivative values
deriv_values = deriv_func(range(1, 1000), *popt_dev)

# Find the first epoch at which the derivative falls below the threshold
threshold = 0.0001
converged_epoch = next((i for i, deriv in enumerate(deriv_values) if abs(deriv) < threshold), len(dev_acc))
print(f"Convergence found at epoch {converged_epoch + warmup}")

test_acc_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-PP-Len{LEN}/Epoch256-LR{LR}-Randomized-Test/eval_results.csv'
	)
test_acc = test_acc_df['eval_accuracy'].values.tolist()[0:-1]
test_acc = list(map(lambda x: x * 100, test_acc))  # convert to percentage
y_test = test_acc[warmup:]
x_test = np.arange(1, len(y_test) + 1)
popt_test, _ = curve_fit(fit_func, x_test, y_test, initial_guess)
print(*popt_test)
converged_test_acc = fit_func(converged_epoch, *popt_test)
print(f"Converged accuracy {converged_test_acc:.4f}.")

# Plot the fitted curve and the original data on the same plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x_dev, y_dev, marker='o', markersize=4, linestyle='-', label='Original data')
ax.plot(x_dev, fitted_dev, color='r', linewidth=2, linestyle='-', label='Fitted curve')
ax.legend(fontsize=12)
ax.set_xlabel('Epoch', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title('Accuracy vs Epoch', fontsize=16)
ax.grid(True)
plt.show()
