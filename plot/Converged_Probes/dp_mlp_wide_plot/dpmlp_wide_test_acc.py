# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator
from scipy.optimize import curve_fit

DIM: int = 768
PROBE_NAME: str = f"DPMLP-Dim{DIM}-Layer8"
# OPT_LR: str = '0.0001'
OPT_LR: str = '5e-05'

# Read data from csv file
pre_test_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR{OPT_LR}-Pretrained-Test/eval_results.csv'
	)
random_test_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR{OPT_LR}-Randomized-Test/eval_results.csv'
	)

# Get the accuracy
pre_test_acc = pre_test_df['eval_accuracy'].values.tolist()[0:-1]
random_test_acc = random_test_df['eval_accuracy'].values.tolist()[0:-1]

# Get the epoch
epoch = pre_test_df['epoch'].values.tolist()[0:-1]


# Fit RLM accuracy
# Define a function to fit to the accuracy values
def fit_func(x, a, b, c):
	# return c - a * x ** (-b)
	return a * np.exp(-b * x) + c


# Compute the derivative of the fitted function
def deriv_func(x, a, b, c):
	# return a * b * x ** (-(b + 1))
	return -a * b * np.exp(-b * x)


# Fit the function to the accuracy values
popt, _ = curve_fit(fit_func, range(len(random_test_acc)), random_test_acc)
print(*popt)

# Compute the fitted curve using the optimal parameters
fitted_curve = fit_func(range(len(random_test_acc)), *popt)

# Compute the derivative values
deriv_values = deriv_func(range(1, len(random_test_acc) + 1), *popt)

# Find the first epoch at which the derivative falls below the threshold
threshold = 0.0001
converged_epoch = next((i for i, deriv in enumerate(deriv_values) if abs(deriv) < threshold), len(random_test_acc))
converged_acc = random_test_acc[converged_epoch]
print(f"Convergence found at epoch {converged_epoch}, with accuracy {converged_acc:.4f}.")



plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.figure(figsize=(8, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
x_major_locator = MultipleLocator(32)
ax.xaxis.set_major_locator(x_major_locator)

# Plot the accuracy
plt.plot(epoch, pre_test_acc, label='PLM', color='salmon', linestyle='-', marker='.')
plt.plot(epoch, random_test_acc, label='RLM', color='skyblue', linestyle='-', marker='.')
plt.plot(epoch, fitted_curve, color='gold', linewidth=2, linestyle='-', label='Fitted RLM')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title(f'{PROBE_NAME}, Optimal LR={OPT_LR}')
plt.legend()
plt.savefig(f'{PROBE_NAME}_test_acc.svg', format='svg', bbox_inches='tight')
plt.show()
