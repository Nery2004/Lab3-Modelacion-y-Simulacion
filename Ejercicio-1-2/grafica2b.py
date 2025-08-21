import numpy as np
import matplotlib.pyplot as plt
from ejercicio1 import gd_random, gd_naive, newton, conjugate_gradient, bfgs


def f_rosenbrock(x):
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def df_rosenbrock(x):
    x1, x2 = x[0], x[1]
    return np.array([
        -400 * x1 * (x2 - x1**2) - 2 * (1 - x1),
        200 * (x2 - x1**2)
    ])

def ddf_rosenbrock(x):
    x1, x2 = x[0], x[1]
    return np.array([
        [-400*(x2 - 3*x1**2)+2, -400*x1],
        [-400*x1, 200]
    ])


x0 = np.array([-1.2, 1.0])
x_star = np.array([1.0, 1.0])

np.random.seed(42)

config = {'alpha': 0.005, 'maxIter': 3000, 'eps': 1e-5}

methods = {}

# Ejecutar métodos
try:
    methods['GD Random'] = gd_random(f_rosenbrock, df_rosenbrock, x0, **config)
except: pass
try:
    methods['GD Naive'] = gd_naive(f_rosenbrock, df_rosenbrock, x0, **config)
except: pass
try:
    methods['Newton'] = newton(f_rosenbrock, df_rosenbrock, ddf_rosenbrock, x0, **config)
except: pass
try:
    methods['Conjugate Gradient'] = conjugate_gradient(f_rosenbrock, df_rosenbrock, x0, **config)
except: pass
try:
    methods['BFGS'] = bfgs(f_rosenbrock, df_rosenbrock, x0, **config)
except: pass


# Visualización 2D de convergencia
x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 100*(Y - X**2)**2 + (1 - X)**2

plt.figure(figsize=(8,6))
plt.contour(X, Y, Z, 50, cmap='viridis')

colors = ['r', 'g', 'b', 'm', 'c']
for i, (name, result) in enumerate(methods.items()):
    xs = np.array(result[2])  # Suponiendo que xs está en la posición 2
    plt.plot(xs[:,0], xs[:,1], label=name, color=colors[i], marker='o', markersize=3)

plt.plot(x_star[0], x_star[1], 'k*', markersize=12, label='x*')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Trayectorias de convergencia - Función Rosenbrock 2D')
plt.legend()
plt.grid(True)
plt.show()
