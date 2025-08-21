import numpy as np
from ejercicio1 import gd_random, gd_naive, newton, conjugate_gradient, bfgs
import matplotlib.pyplot as plt


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

# Configuración del problema
x0 = np.array([-3.0, 1.0])
x_star = np.array([-1.01463, -1.04453])
f_star = -1.51132

print("="*60)
print(f"Función: f(x,y) = x⁴ + y⁴ - 4xy + 0.5y + 1")
print(f"Punto inicial: x₀ = {x0}")
print(f"Óptimo teórico: x* = {x_star}")
print(f"Valor óptimo: f(x*) = {f_star}")

# Verificar la función en el punto inicial
print(f"\nVerificación en el punto inicial:")
print(f"f(x₀) = {f(x0):.6f}")
print(f"||∇f(x₀)|| = {np.linalg.norm(df(x0)):.6f}")
print(f"∇f(x₀) = {df(x0)}")

np.random.seed(42)
methods = {}


try:
    print("Ejecutando GD Random...")
    methods['GD Random'] = gd_random(f, df, x0, alpha=0.01, maxIter=1000, eps=1e-6)
except Exception as e:
    print(f"Error en GD Random: {e}")

try:
    print("Ejecutando GD Naive...")
    methods['GD Naive'] = gd_naive(f, df, x0, alpha=0.01, maxIter=1000, eps=1e-6)
except Exception as e:
    print(f"Error en GD Naive: {e}")

try:
    print("Ejecutando Newton...")
    methods['Newton'] = newton(f, df, ddf, x0, maxIter=1000, eps=1e-6)
except Exception as e:
    print(f"Error en Newton: {e}")

try:
    print("Ejecutando Conjugate Gradient...")
    methods['Conjugate Gradient'] = conjugate_gradient(f, df, x0, maxIter=1000, eps=1e-6)
except Exception as e:
    print(f"Error en Conjugate Gradient: {e}")

try:
    print("Ejecutando BFGS...")
    methods['BFGS'] = bfgs(f, df, x0, maxIter=1000, eps=1e-6)
except Exception as e:
    print(f"Error en BFGS: {e}")


print("RESULTADOS")

for name, result in methods.items():
    print(f"\n{name}:")
    print("-" * len(name))
    
    if len(result) == 6: 
        best, val, xs, fs, errors, converged = result
        iters = len(xs) - 1
    elif len(result) == 7: 
        best, val, xs, fs, errors, converged, iters = result
    else:
        print(f"Formato de resultado inesperado: {len(result)} elementos")
        continue
    
    distance_to_opt = np.linalg.norm(best - x_star)
    value_error = abs(val - f_star)
    final_gradient_norm = np.linalg.norm(df(best))
    
    print(f"  Solución encontrada: x = [{best[0]:.6f}, {best[1]:.6f}]")
    print(f"  Valor de función: f(x) = {val:.6f}")
    print(f"  Convergió: {'✓' if converged else '✗'}")
    print(f"  Iteraciones: {iters}")
    print(f"  ||∇f(x)||: {final_gradient_norm:.2e}")
    print(f"  Distancia al óptimo: ||x - x*|| = {distance_to_opt:.6f}")
    print(f"  Error en valor: |f(x) - f*| = {value_error:.6f}")
    
    if converged:
        if distance_to_opt < 0.01 and value_error < 0.01:
            performance = "EXCELENTE"
        elif distance_to_opt < 0.1 and value_error < 0.1:
            performance = "BUENO"
        elif distance_to_opt < 1.0 and value_error < 1.0:
            performance = "REGULAR"
        else:
            performance = "POBRE "
    else:
        performance = "NO CONVERGIÓ"
    
    print(f"  Evaluación: {performance}")


# Tabla resumen
print(f"\n{'='*100}")
print("TABLA RESUMEN")
print("="*100)
print(f"{'Método':<18} {'Conv':<5} {'Iter':<6} {'f(x)':<12} {'||x-x*||':<10} {'||∇f||':<10} {'Evaluación':<15}")
print("-" * 100)

for name, result in methods.items():
    if len(result) == 6:
        best, val, xs, fs, errors, converged = result
        iters = len(xs) - 1
    elif len(result) == 7:
        best, val, xs, fs, errors, converged, iters = result
    else:
        continue
    
    distance_to_opt = np.linalg.norm(best - x_star)
    final_gradient_norm = np.linalg.norm(df(best))
    
    conv_str = "✓" if converged else "✗"
    
    if converged and distance_to_opt < 0.01:
        eval_str = "EXCELENTE"
    elif converged and distance_to_opt < 0.1:
        eval_str = "BUENO"
    elif converged:
        eval_str = "REGULAR"
    else:
        eval_str = "NO CONV"
    
    print(f"{name:<18} {conv_str:<5} {iters:<6d} {val:<12.6f} {distance_to_opt:<10.6f} "
          f"{final_gradient_norm:<10.2e} {eval_str:<15}")


# Gráfica de convergencia
plt.figure(figsize=(10,6))
for name, result in methods.items():
    if len(result) == 6:
        _, _, xs, fs, errors, _ = result
    elif len(result) == 7:
        _, _, xs, fs, errors, _, _ = result
    else:
        continue
    plt.plot(errors, label=name)

plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Error de aproximación")
plt.title("Comparación de convergencia en los 5 métodos")
plt.legend()
plt.grid(True)
plt.show()
