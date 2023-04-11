# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

PROBE_NAME: str = f"DPMLP-Dim512-Layer4"


plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

# Read data from csv file
rand_lr5e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0005-Randomized-Dev/eval_results.csv'
	)
rand_lr1e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0001-Randomized-Dev/eval_results.csv'
	)
rand_lr5e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-05-Randomized-Dev/eval_results.csv'
	)
rand_lr1e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR1e-05-Randomized-Dev/eval_results.csv'
	)
rand_lr5e_6_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-06-Randomized-Dev/eval_results.csv'
	)

pre_lr5e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0005-Pretrained-Dev/eval_results.csv'
	)
pre_lr1e_4_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR0.0001-Pretrained-Dev/eval_results.csv'
	)
pre_lr5e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-05-Pretrained-Dev/eval_results.csv'
	)
pre_lr1e_5_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR1e-05-Pretrained-Dev/eval_results.csv'
	)
pre_lr5e_6_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}/Epoch256-LR5e-06-Pretrained-Dev/eval_results.csv'
	)


# Get the accuracy
rand_lr5e_4_acc = rand_lr5e_4_df['eval_accuracy'].values.tolist()[0:-1]
rand_lr1e_4_acc = rand_lr1e_4_df['eval_accuracy'].values.tolist()[0:-1]
rand_lr5e_5_acc = rand_lr5e_5_df['eval_accuracy'].values.tolist()[0:-1]
rand_lr1e_5_acc = rand_lr1e_5_df['eval_accuracy'].values.tolist()[0:-1]
rand_lr5e_6_acc = rand_lr5e_6_df['eval_accuracy'].values.tolist()[0:-1]

pre_lr5e_4_acc = pre_lr5e_4_df['eval_accuracy'].values.tolist()[0:-1]
pre_lr1e_4_acc = pre_lr1e_4_df['eval_accuracy'].values.tolist()[0:-1]
pre_lr5e_5_acc = pre_lr5e_5_df['eval_accuracy'].values.tolist()[0:-1]
pre_lr1e_5_acc = pre_lr1e_5_df['eval_accuracy'].values.tolist()[0:-1]
pre_lr5e_6_acc = pre_lr5e_6_df['eval_accuracy'].values.tolist()[0:-1]

# Get the epoch
epoch = rand_lr1e_4_df['epoch'].values.tolist()[0:-1]



plt.figure(figsize=(8, 5))
plt.grid(linestyle="--")  # 设置背景网格线为虚线
ax = plt.gca()
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
x_major_locator = MultipleLocator(32)
ax.xaxis.set_major_locator(x_major_locator)

# Plot the accuracy
plt.plot(epoch, rand_lr5e_4_acc, label='Rand, lr=5e-4', color='limegreen', linestyle='-', marker='.')
plt.plot(epoch, rand_lr1e_4_acc, label='Rand, lr=1e-4', color='seagreen', linestyle='-', marker='.')
plt.plot(epoch, rand_lr5e_5_acc, label='Rand, lr=5e-5', color='skyblue', linestyle='-', marker='.')
plt.plot(epoch, rand_lr1e_5_acc, label='Rand, lr=1e-5', color='royalblue', linestyle='-', marker='.')
plt.plot(epoch, rand_lr5e_6_acc, label='Rand, lr=5e-6', color='slateblue', linestyle='-', marker='.')

plt.plot(epoch, pre_lr5e_4_acc, label='Pre, lr=5e-4', color='lightcoral', linestyle='-', marker='.')
plt.plot(epoch, pre_lr1e_4_acc, label='Pre, lr=1e-4', color='firebrick', linestyle='-', marker='.')
plt.plot(epoch, pre_lr5e_5_acc, label='Pre, lr=5e-5', color='orangered', linestyle='-', marker='.')
plt.plot(epoch, pre_lr1e_5_acc, label='Pre, lr=1e-5', color='orange', linestyle='-', marker='.')
plt.plot(epoch, pre_lr5e_6_acc, label='Pre, lr=5e-6', color='gold', linestyle='-', marker='.')


plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Validation Accuracy of {PROBE_NAME} vs Epoch vs Learning Rates')
plt.legend(ncol=2, loc='lower right')
plt.savefig(f'Special-{PROBE_NAME}_epoch_acc_lr.svg', format='svg', bbox_inches='tight')
plt.show()
