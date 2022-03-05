import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

steps = 720
x = np.arange(0, 250*steps, dtype=float)
y = np.zeros_like(x)
lr = 0.001
rate = math.exp(math.log(1e-5/lr)*(steps/250/720))

print('rate: {}'.format(rate))

it = np.nditer(x, flags=['f_index'])
while not it.finished:
    y[it.index] = lr * pow(rate, math.floor(it[0]/steps))
    it.iternext()

plt.plot(x, y)
plt.show()

