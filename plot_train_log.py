import numpy as np
import matplotlib.pyplot as plt

loss = []
prec = []
with open("train_log.txt") as f:
    lines = f.readlines()
    for l in lines:
        s = l[11:19]
        pos = l.find("precision : ")
        p = l[pos+13: pos+19]
        loss.append(float(s))
        prec.append(float(p))
# print(prec)
loss = np.array(loss)
prec = np.array(prec)
display_len = 1000
plt.plot(np.arange(display_len), np.array(loss[0:display_len]),  c='y', alpha=0.5)
plt.plot(np.arange(display_len), np.array(prec[0:display_len]), c='r', alpha=0.5)
plt.legend(['loss','precision'])
plt.title('Training')
plt.show()