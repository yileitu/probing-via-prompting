# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

DIM: int = 4096
PROBE_NAME: str = f"DPMLP-Dim{DIM}-Layer8"

# Read data from csv file
lr1e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0001-Randomized-Dev/eval_results.csv'
	)
lr5e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-05-Randomized-Dev/eval_results.csv'
	)
lr1e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR1e-05-Randomized-Dev/eval_results.csv'
	)
lr5e_6_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-06-Randomized-Dev/eval_results.csv'
	)

# Get the accuracy
lr1e_4_acc = lr1e_4_df['eval_accuracy'].values.tolist()[0:-1]
lr5e_5_acc = lr5e_5_df['eval_accuracy'].values.tolist()[0:-1]
lr1e_5_acc = lr1e_5_df['eval_accuracy'].values.tolist()[0:-1]
lr5e_6_acc = lr5e_6_df['eval_accuracy'].values.tolist()[0:-1]

# Get the epoch
epoch = lr1e_4_df['epoch'].values.tolist()[0:-1]

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
plt.plot(epoch, lr1e_4_acc, label='lr=1e-4', color='salmon', linestyle='-', marker='.')
plt.plot(epoch, lr5e_5_acc, label='lr=5e-5', color='skyblue', linestyle='-', marker='.')
plt.plot(epoch, lr1e_5_acc, label='lr=1e-5', color='gold', linestyle='-', marker='.')
plt.plot(epoch, lr5e_6_acc, label='lr=5e-6', color='limegreen', linestyle='-', marker='.')

PLOT_PRETRAINED: bool = False
if PLOT_PRETRAINED:
	PRETRAINED_LR: str = '5e-05'
	pretrained_df = pd.read_csv(
		f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/Wide/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR{PRETRAINED_LR}-Pretrained-Dev/eval_results.csv'
		)
	pretrained_acc = pretrained_df['eval_accuracy'].values.tolist()[0:-1]
	plt.plot(
		epoch, pretrained_acc, label=f'Pretrained, lr={PRETRAINED_LR}', color='slateblue', linestyle='-', marker='.'
		)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.title(f'{PROBE_NAME}, RLM')
plt.legend()
if not PLOT_PRETRAINED:
	plt.savefig(f'{PROBE_NAME}_val_acc.svg', format='svg', bbox_inches='tight')
else:
	plt.savefig(f'{PROBE_NAME}_pretrained_epoch_acc_lr.svg', format='svg', bbox_inches='tight')
plt.show()
