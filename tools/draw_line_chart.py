import matplotlib.pyplot as plt
import numpy as np

# 创建一些数据
x1 = [15.7, 14.4, 14.5]
y1 = [83.1, 82.86, 81.72]

x2 = [22.5, 19.2, 21.4]
y2 = [94.41, 93.41, 92.91]


# 选择两个点
point1 = (x1[0], y1[0])
point2 = (x2[0], y2[0])

# 创建一个新的图形
plt.figure()

# 绘制散点图
plt.scatter(x1, y1)
plt.scatter(x2, y2)

# 使用annotate添加箭头线
plt.annotate("",
             xy=point2, xycoords='data',
             xytext=point1, textcoords='data',
             arrowprops=dict(arrowstyle="->",
                             connectionstyle="arc3"),
             )

# 显示图形
plt.show()