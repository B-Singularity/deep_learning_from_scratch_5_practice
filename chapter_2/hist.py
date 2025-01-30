import os
import numpy as np
import matplotlib.pyplot as plt

path = os.path.join(os.path.dirname(__file__), 'height.txt')
xs = np.loadtxt(path)
print(xs.shape)

mu = np.mean(xs)
sigma = np.std(xs)

print(mu)
print(sigma)

# plt.hist(xs, bins='auto', density=True)
# plt.xlabel("Height(cm)")
# plt.ylabel("Probability")
# plt.show()