# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

df = pd.read_excel("/Users/tuyilei/Desktop/NLP_SP/EACL_submission/sheets/DP-MLP-Deep.xlsx")

depths = df["Layer"].astype(int).tolist()  # 深度值列表
pretrained_accuracies = df["acc_test_plm"].astype(float).tolist()  # 对应的pretrained精确度列表
randomized_accuracies = df["acc_test_rlm"].astype(float).tolist()  # 对应的randomized精确度列表

# 计算两个精度之间的差值
accuracy_differences = [p - r for p, r in zip(pretrained_accuracies, randomized_accuracies)]

# 绘制pretrained accuracy和randomized accuracy
FONT_SIZE = 15
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(5.1, 4))
plt.plot(depths, pretrained_accuracies, label='PLM', color='#E74C3C')
plt.plot(depths, randomized_accuracies, label='RLM', color='#F1C40F')

# 标记每个数据点
plt.scatter(depths, pretrained_accuracies, color='#E74C3C')
plt.scatter(depths, randomized_accuracies, color='#F1C40F')

# 在每个点上画出差值
idx: int = 0
for d, p, r, diff in zip(depths, pretrained_accuracies, randomized_accuracies, accuracy_differences):
	if idx == 0:
		mid = (p + r) / 2
		plt.plot([d, d], [p, r], color='#3498DB', linestyle='--', linewidth=0.5)  # 绘制蓝色连线
		plt.text(d, mid + 0.05, f'{diff:.3f}', color='black', fontsize=FONT_SIZE, ha='center')  # 黑色的差值标签
	elif idx == 1:
		mid = (p + r) / 2
		plt.plot([d, d], [p, r], color='#3498DB', linestyle='--', linewidth=0.5)  # 绘制蓝色连线
		plt.text(d, mid + 0.02, f'{diff:.3f}', color='black', fontsize=FONT_SIZE, ha='center')  # 黑色的差值标签
	elif idx == 3:
		mid = (p + r) / 2
		plt.plot([d, d], [p, r], color='#3498DB', linestyle='--', linewidth=0.5)  # 绘制蓝色连线
		plt.text(d, mid - 0.02, f'{diff:.3f}', color='black', fontsize=FONT_SIZE, ha='center')  # 黑色的差值标签
	else:
		mid = (p + r) / 2
		plt.plot([d, d], [p, r], color='#3498DB', linestyle='--', linewidth=0.5)  # 绘制蓝色连线
		plt.text(d, mid, f'{diff:.3f}', color='black', fontsize=FONT_SIZE, ha='center')  # 黑色的差值标签
	idx += 1

# 使用淡蓝色在两条线之间填充颜色
plt.fill_between(depths, pretrained_accuracies, randomized_accuracies, color='#D6EAF8', alpha=0.5)

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.subplots_adjust(top=0.92)  # 调整标题位置
plt.suptitle('DP-MLP (Deep)', fontsize=FONT_SIZE)
plt.xlabel('#HiddenLayer (Depth)', fontsize=FONT_SIZE)
plt.ylabel('Converged Test Accuracy', fontsize=FONT_SIZE)
plt.savefig('dp_deep_acc_comparison.svg', format='svg', bbox_inches='tight')
plt.show()
