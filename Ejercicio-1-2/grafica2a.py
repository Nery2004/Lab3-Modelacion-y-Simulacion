import numpy as np
import matplotlib.pyplot as plt
from ejercicio1 import gd_random, gd_naive, newton, conjugate_gradient, bfgs


def f(x):
    x_val, y_val = x[0], x[1]
    return x_val**4 + y_val**4 - 4*x_val*y_val + 0.5*y_val + 1

def df(x):
    x_val, y_val = x[0], x[1]
    return np.array([
        4*x_val**3 - 4*y_val,
        4*y_val**3 - 4*x_val + 0.5
    ])

def ddf(x):
    x_val, y_val = x[0], x[1]
    return np.array([
        [12*x_val**2, -4],
        [-4, 12*y_val**2]
    ])


x0 = np.array([-3.0, 1.0])
x_star = np.array([-1.01463, -1.04453])

np.random.seed(42)
config = {'alpha': 0.01, 'maxIter': 1000, 'eps': 1e-6}

methods = {}

# Ejecutar métodos
try:
    methods['GD Random'] = gd_random(f, df, x0, **config)
except: pass
try:
    methods['GD Naive'] = gd_naive(f, df, x0, **config)
except: pass
try:
    methods['Newton'] = newton(f, df, ddf, x0, **config)
except: pass
try:
    methods['Conjugate Gradient'] = conjugate_gradient(f, df, x0, **config)
except: pass
try:
    methods['BFGS'] = bfgs(f, df, x0, **config)
except: pass


# Visualización 2D de la convergencia
x_vals = np.linspace(-3.5, 3.5, 200)
y_vals = np.linspace(-3.5, 3.5, 200)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**4 + Y**4 - 4*X*Y + 0.5*Y + 1

plt.figure(figsize=(8,6))
plt.contour(X, Y, Z, 50, cmap='viridis')

colors = ['r', 'g', 'b', 'm', 'c']
for i, (name, result) in enumerate(methods.items()):
    xs = np.array(result[2])  # Suponiendo que xs está en la posición 2
    plt.plot(xs[:,0], xs[:,1], label=name, color=colors[i], marker='o', markersize=3)

plt.plot(x_star[0], x_star[1], 'k*', markersize=10, label='x*')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trayectorias de convergencia (2D)')
plt.legend()
plt.grid(True)
plt.show()
