import numpy as np
import matplotlib.pyplot as plt

size = 3
a = [54.1375,54.1375]
b = [22.4806,18.9571]
c = [32.6821,27.6122]
x = np.arange(size)
total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label="Method1")
plt.bar(x + width, b, width=width, label="Method2")

x_labels = ['diginetica','Nowplaying','Tmall']
plt.xticks(x, x_labels)
# 用第1组...替换横坐标x的值
for i, j in zip(x, a):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
for i, j in zip(x + width, b):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)
for i, j in zip(x + 2 * width, c):
    plt.text(i, j + 0.01, "%.2f" % j, ha="center", va="bottom", fontsize=7)

# 显示图例
plt.legend()
# 显示柱状图
plt.show()