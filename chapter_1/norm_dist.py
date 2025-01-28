import numpy as np
import matplotlib.pyplot as plt

def normal(x, mu=0, sigma=1):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

# x = np.linspace(-5, 5, 100)
# y = normal(x)
#
# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# x = np.linspace(-10, 10, 1000)
# y0 = normal(x, mu=0, sigma=0.5)
# y1 = normal(x, mu=0, sigma=1)
# y2 = normal(x, mu=0, sigma=2)
#
# plt.plot(x, y0, label='$\mu$=-3')
# plt.plot(x, y1,  label='$\mu$=0')
# plt.plot(x, y2, label='$\mu$=5')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()


x_means = []
N = 10

for _ in range(10000):
    xs = []
    for n in range(N):
        x = np.random.rand()
        xs.append(x)
    mean = np.mean(xs)
    x_means.append(mean)

plt.hist(x_means, bins='auto', density=True)
plt.title(f' N = {N}')
plt.xlabel('x')
plt.ylabel('probability density')
plt.xlim(-0.05, 1.05)
plt.ylim(0, 5)
plt.show()

