import numpy as np

# 转移概率矩阵
a = np.array([[0.5, 0.1, 0.4], [0.3, 0.5, 0.2], [0.2, 0.2, 0.6]])
# 观测概率矩阵
b = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])
# 初始概率分布
p = np.array([0.2, 0.3, 0.5])
# 已知的观测序列
o = np.array([0, 1, 0, 0, 1, 0, 1, 1])
# 时间阶段数
t = 8
# 状态数
N = a.shape[0]


def back(obj_t):
    # t时刻后向概率
    beta_t = np.ones(N)
    # t-1时刻后向概率
    beta_t_1 = np.zeros(N)
    # 递推求t=obj_t时刻的各状态后向概率
    for i in range(t - obj_t):
        for j in range(N):
            observe_index = o[t - 1 - i]
            beta_t_1[j] = sum(a[j, :] * b[:, observe_index] * beta_t)
        beta_t = beta_t_1.copy()
    return beta_t


def forward(obj_t):
    # t-1时刻前向概率
    alpha_t_1 = np.zeros(N)
    # 初始化alpha_t_1
    observe_index = o[0]
    alpha_t_1 = p * b[:, observe_index]
    # t時刻前向概率
    alpha_t = np.zeros(N)
    for i in range(obj_t - 1):
        for j in range(N):
            observe_index = o[i + 1]
            alpha_t[j] = alpha_t_1.dot(a[:, j]) * b[j][observe_index]
        alpha_t_1 = alpha_t.copy()
    return alpha_t


if __name__ == '__main__':
    obj_t = 4
    obj_state = 3
    prob = back(obj_t)[obj_state - 1] * forward(obj_t)[obj_state - 1] / back(obj_t).dot(forward(obj_t))
    print(prob)
