import numpy as np

# 转移概率矩阵
a = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])
# 观测概率矩阵
b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
# 初始概率分布
p = np.array([0.2, 0.4, 0.4])
# 已知的观测序列
o = np.array([0, 1, 0, 1])
# 时间阶段数
t = 4
# 目标概率
prob = 0
# 状态数
N = a.shape[0]
# t时刻后向概率
beta_t = np.ones(N)
# t-1时刻后向概率
beta_t_1 = np.zeros(N)
# 递推求t=1时刻的各状态后向概率
for i in range(t - 1):
    for j in range(N):
        observe_index = o[t - 1 - i]
        beta_t_1[j] = sum(a[j, :] * b[:, observe_index] * beta_t)
    beta_t = beta_t_1.copy()
# 求和得到目标观测序列的概率
observe_index = o[0]
prob = sum(p * b[:, observe_index] * beta_t)
print(prob)
