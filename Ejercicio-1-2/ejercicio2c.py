import numpy as np
from ejercicio1 import gd_random, gd_naive, newton, conjugate_gradient, bfgs

# Función de Rosenbrock 7D y derivadas
def f_rosenbrock(x):
    total = 0.0
    for i in range(6):
        total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return total

def df_rosenbrock(x):
    grad = np.zeros_like(x)
    n = len(x)
    for i in range(n):
        if i < n-1:
            grad[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
        if i > 0:
            grad[i] += 200*(x[i] - x[i-1]**2)
    return grad

def ddf_rosenbrock(x):
    n = len(x)
    H = np.zeros((n,n))
    for i in range(n):
        if i < n-1:
            H[i,i] += 1200*x[i]**2 - 400*x[i+1] + 2
            H[i,i+1] += -400*x[i]
            H[i+1,i] += -400*x[i]
            H[i+1,i+1] += 200
    return H

x0 = np.array([-1.2, 1, 1, 1, 1, -1.2, 1])
x_star = np.ones(7)
f_star = 0.0

print("FUNCIÓN ROSENBROCK 7D")
print(f"Punto inicial: x₀ = {x0}")
print(f"Óptimo teórico: x* = {x_star}")
print(f"Valor óptimo: f(x*) = {f_star}")

print(f"\nVerificación en el punto inicial:")
print(f"f(x₀) = {f_rosenbrock(x0):.6f}")
print(f"||∇f(x₀)|| = {np.linalg.norm(df_rosenbrock(x0)):.6f}")

print(f"\nVerificación en el óptimo teórico:")
print(f"f(x*) = {f_rosenbrock(x_star):.6f}")
print(f"||∇f(x*)|| = {np.linalg.norm(df_rosenbrock(x_star)):.2e}")

configs = {
    'balanced': {
        'alpha': 0.001,
        'maxIter': 10000,
        'eps': 1e-6,
        'description': 'Configuración balanceada para 7D'
    }
}
config = configs['balanced']

print(f"\nUsando configuración: {config['description']}")
print(f"α = {config['alpha']}, maxIter = {config['maxIter']}, ε = {config['eps']}")

np.random.seed(42)

methods = {}

print(f"\n{'='*60}")
print("EJECUTANDO MÉTODOS...")
print("="*60)

try:
    print("Ejecutando GD Random...")
    methods['GD Random'] = gd_random(f_rosenbrock, df_rosenbrock, x0, 
                                    alpha=config['alpha'], maxIter=config['maxIter'], 
                                    eps=config['eps'])
except Exception as e:
    print(f"Error en GD Random: {e}")

try:
    print("Ejecutando GD Naive...")
    methods['GD Naive'] = gd_naive(f_rosenbrock, df_rosenbrock, x0, 
                                  alpha=config['alpha'], maxIter=config['maxIter'], 
                                  eps=config['eps'])
except Exception as e:
    print(f"Error en GD Naive: {e}")

try:
    print("Ejecutando Newton...")
    methods['Newton'] = newton(f_rosenbrock, df_rosenbrock, ddf_rosenbrock, x0, 
                              maxIter=config['maxIter'], eps=config['eps'])
except Exception as e:
    print(f"Error en Newton: {e}")

try:
    print("Ejecutando Conjugate Gradient...")
    methods['Conjugate Gradient'] = conjugate_gradient(f_rosenbrock, df_rosenbrock, x0, 
                                                      maxIter=config['maxIter'], eps=config['eps'])
except Exception as e:
    print(f"Error en Conjugate Gradient: {e}")

try:
    print("Ejecutando BFGS...")
    methods['BFGS'] = bfgs(f_rosenbrock, df_rosenbrock, x0, 
                          maxIter=config['maxIter'], eps=config['eps'])
except Exception as e:
    print(f"Error en BFGS: {e}")

for name, result in methods.items():
    print(f"\n{name}:")
    print("-" * len(name))
    
    if len(result) == 6:
        best, val, xs, fs, errors, converged = result
        iters = len(xs) - 1
    elif len(result) == 7:
        best, val, xs, fs, errors, converged, iters = result
    else:
        print(f"Formato inesperado: {len(result)} elementos")
        continue
    
    distance_to_opt = np.linalg.norm(best - x_star)
    value_error = abs(val - f_star)
    final_gradient_norm = np.linalg.norm(df_rosenbrock(best))
    
    print(f"  Solución encontrada: x = {best}")
    print(f"  Valor de función: f(x) = {val:.6f}")
    print(f"  Convergió: {'✓' if converged else '✗'}")
    print(f"  Iteraciones: {iters}")
    print(f"  ||∇f(x)||: {final_gradient_norm:.2e}")
    print(f"  Distancia al óptimo: ||x - x*|| = {distance_to_opt:.6f}")
    print(f"  Error en valor: |f(x) - f*| = {value_error:.6f}")

print("="*50)
print("TABLA RESUMEN - FUNCIÓN ROSENBROCK 7D")
print("="*50)
print(f"{'Método':<18} {'Conv':<5} {'Iter':<6} {'f(x)':<12} {'||x-x*||':<10} {'||∇f||':<10}")

for name, result in methods.items():
    if len(result) == 6:
        best, val, xs, fs, errors, converged = result
        iters = len(xs) - 1
    elif len(result) == 7:
        best, val, xs, fs, errors, converged, iters = result
    else:
        continue
    
    distance_to_opt = np.linalg.norm(best - x_star)
    final_gradient_norm = np.linalg.norm(df_rosenbrock(best))
    conv_str = "✓" if converged else "✗"
    
    print(f"{name:<18} {conv_str:<5} {iters:<6d} {val:<12.6f} {distance_to_opt:<10.6f} {final_gradient_norm:<10.2e}")


# Gráfica comparativa de errores
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for name, result in methods.items():
    if len(result) == 6:
        best, val, xs, fs, errors, converged = result
    elif len(result) == 7:
        best, val, xs, fs, errors, converged, iters = result
    else:
        continue
    
    plt.plot(errors, label=name)

plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.title("Evolución del error en Rosenbrock 7D")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()
