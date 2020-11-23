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
# 状态数
N = a.shape[0]
# t时刻后向概率
beta_t = np.ones(N)
# t-1时刻后向概率
beta_t_1 = np.zeros(N)


# 递推求t=1时刻的各状态后向概率

def viterbi():
    # t时刻为i状态，且满足o1,o2,...ot的最大概率路径的概率
    delta_t = np.zeros(N)
    # 初始化delta_t
    observe_index = o[0]
    delta_t = p * b[:, observe_index]
    # t+1时刻为i状态，且满足o1,o2,...ot+1的最大概率路径的概率
    delta_t_1 = np.zeros(N)
    # t时刻为i状态的最大概率路径的t-1个点  矩阵
    psi = np.zeros((N, t))

    for i in range(t - 1):
        for j in range(N):
            observe_index = o[i + 1]
            index = np.argmax(delta_t * a[:, j])
            psi[j][i + 1] = index
            delta_t_1[j] = (delta_t * a[:, j])[index] * b[j][observe_index]
        delta_t = delta_t_1.copy()
    i_t = np.argmax(delta_t)
    prob_max = max(delta_t)
    return prob_max, i_t, psi


def back_road(psi, i_t):
    road = []
    for i in range(t):
        road.append(i_t+1)
        i_t = int(psi[i_t, t - i - 1])
    return road


if __name__ == '__main__':
    prob_max, i_t, psi = viterbi()
    print(psi)
    print('出现目标观测序列的最大概率：', prob_max)
    print('出现目标观测序列的末状态：', i_t+1)
    road=back_road(psi, i_t)
    road.reverse()
    print('出现目标观测序列的最优路径：', road)

