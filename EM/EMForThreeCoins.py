import matplotlib.pyplot as plt

t = [0.46]
p = [0.55]
q = [0.67]
y = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
B_up = []
for i in range(1000):
    for j in range(len(y)):
        temp = (t[i] * (p[i] ** y[j]) * (1 - p[i]) ** (1 - y[j])) / \
               ((t[i] * (p[i] ** y[j]) * (1 - p[i]) ** (1 - y[j])) +
                ((1 - t[i]) * (q[i] ** y[j]) * (1 - q[i]) ** (1 - y[j])))
        B_up.append(temp)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    # print(len(B_up))
    for k in range(len(B_up)):
        sum1 += B_up[k]
        sum2 += B_up[k] * y[k]
        sum3 += (1 - B_up[k])
        sum4 += (1 - B_up[k]) * y[k]
    t.append(sum1 / len(y))
    p.append(sum2 / sum1)
    q.append(sum4 / sum3)
    B_up.clear()

plt.plot(range(len(t)), t, label="t")
plt.plot(range(len(p)), p, label="p")
plt.plot(range(len(q)), q, label="q")
plt.legend()
plt.show()
print("t:" + str(t[-1]))
print("p:" + str(p[-1]))
print("q:" + str(q[-1]))
