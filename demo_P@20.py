import numpy as np
import matplotlib.pyplot as plt

size = 3
a = [54.1375,49.8455,53.9929]
b = [22.4806,20.1639,23.2889]
c = [32.6821,35.2267,33.2921]
x = np.arange(size)
total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

plt.bar(x, a, width=width, label="Global and Local+Long and Short Term Interest")
plt.bar(x + width, b, width=width, label="Global and Local")
plt.bar(x + 2*width, c, width=width, label="Long and Short Term Interest")

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
