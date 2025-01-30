import numpy as np
import matplotlib.pyplot as plt

# xs = np.arange(-2, 2, 0.1)
# ys = np.arange(-2, 2, 0.1)
#
# X, Y = np.meshgrid(xs, ys)
# Z = X ** 2 + Y ** 2
#
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2

ax = plt.axes()
ax.contour(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()