from math import sqrt

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

epoch = 1000
K = 2
a = np.zeros((K, epoch + 1))
b = np.zeros((K, epoch + 1))
c = np.zeros((K, epoch + 1))
# a[:, 0] = 0.5
a[0][0]=0.43
a[1][0]=0.57
# b[:, 0] = 10
# c[:, 0] = 10
b[0][0]=20
b[1][0]=45
c[0][0]=200
c[1][0]=100
y = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
r = np.zeros((K, len(y)))
for i in range(epoch):
    for j in range(len(y)):
        for k in range(K):
            temp = 0
            for m in range(K):
                temp += a[m][i] * st.norm.pdf(y[j], b[m][i], sqrt(c[m][i]))
            r[k][j] = a[k][i] * st.norm.pdf(y[j], b[k][i], sqrt(c[k][i])) / temp
    sum1 = [0] * K
    sum2 = [0] * K
    sum3 = [0] * K
    for k in range(K):
        for j in range(len(y)):
            sum1[k] += r[k][j] * y[j]
            sum2[k] += r[k][j]
        a[k][i + 1] = sum2[k] / len(y)
        b[k][i + 1] = sum1[k] / sum2[k]
    for k in range(K):
        for j in range(len(y)):
            sum3[k] += r[k][j] * (y[j] - b[k][i + 1]) ** 2
        c[k][i + 1] = sum3[k] / sum2[k]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# matplotlib画图中中文显示会有问题，需要这两行设置默认字体
plt.plot(range(epoch + 1), a[0], label="高斯分量1系数")
plt.plot(range(epoch + 1), a[1], label="高斯分量2系数")
plt.plot(range(epoch + 1), b[0], label="高斯分量1均值")
plt.plot(range(epoch + 1), b[1], label="高斯分量2均值")
plt.plot(range(epoch + 1), c[0], label="高斯分量1方差")
plt.plot(range(epoch + 1), c[1], label="高斯分量2方差")
plt.legend()
plt.show()
print(a)
print(b)
print(c)
# print(st.norm.pdf(0))
# print(1/sqrt(2*pi))
