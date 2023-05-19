# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

LEN: int = 100
PROBE_NAME: str = f"PP-flat-Len{LEN}"

# Read data from csv file
lr1e_2_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.01-Randomized-Dev/eval_results.csv'
	)
lr5e_3_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.005-Randomized-Dev/eval_results.csv'
	)
lr1e_3_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.001-Randomized-Dev/eval_results.csv'
	)
lr5e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/pp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0005-Randomized-Dev/eval_results.csv'
	)

# Get the accuracy
lr1e_2_acc = lr1e_2_df['eval_accuracy'].values.tolist()[0:-1]
lr5e_3_acc = lr5e_3_df['eval_accuracy'].values.tolist()[0:-1]
lr1e_3_acc = lr1e_3_df['eval_accuracy'].values.tolist()[0:-1]
lr5e_4_acc = lr5e_4_df['eval_accuracy'].values.tolist()[0:-1]


# Get the epoch
epoch_1e_2 = lr1e_2_df['epoch'].values.tolist()[0:-1]
epoch_5e_3 = lr5e_3_df['epoch'].values.tolist()[0:-1]
epoch_1e_3 = lr1e_3_df['epoch'].values.tolist()[0:-1]
epoch_5e_4 = lr5e_4_df['epoch'].values.tolist()[0:-1]

epoch = min(epoch_1e_2, epoch_5e_3, epoch_1e_3, epoch_5e_4)
lr1e_2_acc = lr1e_2_acc[0:len(epoch)]
lr5e_3_acc = lr5e_3_acc[0:len(epoch)]
lr1e_3_acc = lr1e_3_acc[0:len(epoch)]
lr5e_4_acc = lr5e_4_acc[0:len(epoch)]


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
plt.plot(epoch, lr1e_2_acc, label='lr=1e-2', color='skyblue', linestyle='-', marker='.')
plt.plot(epoch, lr5e_3_acc, label='lr=5e-3', color='salmon', linestyle='-', marker='.')
plt.plot(epoch, lr1e_3_acc, label='lr=1e-3', color='gold', linestyle='-', marker='.')
plt.plot(epoch, lr5e_4_acc, label='lr=5e-4', color='limegreen', linestyle='-', marker='.')

PLOT_PRETRAINED: bool = False
if PLOT_PRETRAINED:
	PRETRAINED_LR: str = '5e-05'
	pretrained_df = pd.read_csv(
		f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR{PRETRAINED_LR}-Pretrained-Dev/eval_results.csv'
		)
	pretrained_acc = pretrained_df['eval_accuracy'].values.tolist()[0:-1]
	plt.plot(
		epoch, pretrained_acc, label=f'Pretrained, lr={PRETRAINED_LR}', color='slateblue', linestyle='-', marker='.'
		)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Evaluation Accuracy of {PROBE_NAME} vs Epoch vs Learning Rates')
plt.legend()
if not PLOT_PRETRAINED:
	plt.savefig(f'{PROBE_NAME}_epoch_acc_lr.svg', format='svg', bbox_inches='tight')
else:
	plt.savefig(f'{PROBE_NAME}_pretrained_epoch_acc_lr.svg', format='svg', bbox_inches='tight')
plt.show()
