# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

DIM: int = 512
LAYER: int = 8
PROBE_NAME: str = f"DPMLP-Dim{DIM}-Layer{LAYER}"
LR: str = "0.0001"
# LR: str = "5e-05"

# Read data from csv file
onehot_acc_df = pd.read_csv(
	f'/Users/tuyilei/Desktop/NLP_SP/probing-via-prompting/outputs/dp/mlp/ner/ConvergedProbe-ner-{PROBE_NAME}-OneHot/Epoch256-LR{LR}-Randomized-Test/eval_results.csv'
	)

# Get the accuracy
onehot_acc = onehot_acc_df['eval_accuracy'].values.tolist()[0:-1]

# Get the epoch
epoch = onehot_acc_df['epoch'].values.tolist()[0:-1]

# Plot the accuracy
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
plt.plot(epoch, onehot_acc, label=f'lr={LR}', color='skyblue', linestyle='-', marker='.')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'Onehot Evaluation Accuracy of {PROBE_NAME} with LR{LR} vs Epoch ')
plt.legend(loc='lower right')
plt.savefig(f'{PROBE_NAME}_onehot_epoch_acc.svg', format='svg', bbox_inches='tight')
plt.show()
