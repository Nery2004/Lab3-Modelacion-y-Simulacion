import numpy as np
import matplotlib.pyplot as plt
from ejercicio1 import gd_random, gd_naive, newton, conjugate_gradient, bfgs


# Función de Rosenbrock 2D y derivadas
def f_rosenbrock(x):
    """
    Función de Rosenbrock 2D: f(x1,x2) = 100(x2 - x1²)² + (1 - x1)²
    """
    x1, x2 = x[0], x[1]
    return 100 * (x2 - x1**2)**2 + (1 - x1)**2

def df_rosenbrock(x):
    """
    Gradiente de la función de Rosenbrock 2D:
    ∂f/∂x1 = -400*x1*(x2 - x1²) - 2*(1 - x1)
    ∂f/∂x2 = 200*(x2 - x1²)
    """
    x1, x2 = x[0], x[1]
    df_dx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df_dx2 = 200 * (x2 - x1**2)
    return np.array([df_dx1, df_dx2])

def ddf_rosenbrock(x):
    """
    Hessiano de la función de Rosenbrock 2D:
    ∂²f/∂x1² = -400*(x2 - 3*x1²) + 2
    ∂²f/∂x1∂x2 = -400*x1
    ∂²f/∂x2² = 200
    """
    x1, x2 = x[0], x[1]
    d2f_dx1dx1 = -400 * (x2 - 3 * x1**2) + 2
    d2f_dx1dx2 = -400 * x1
    d2f_dx2dx2 = 200
    
    return np.array([
        [d2f_dx1dx1, d2f_dx1dx2],
        [d2f_dx1dx2, d2f_dx2dx2]
    ])


# Configuración del problema
x0 = np.array([-1.2, 1.0])
x_star = np.array([1.0, 1.0])
f_star = 0.0

print("FUNCIÓN ROSENBROCK")
print(f"Función: f(x1,x2) = 100(x2 - x1²)² + (1 - x1)²")
print(f"Punto inicial: x₀ = {x0}")
print(f"Óptimo teórico: x* = {x_star}")
print(f"Valor óptimo: f(x*) = {f_star}")

# Verificar la función en el punto inicial
print(f"\nVerificación en el punto inicial:")
print(f"f(x₀) = {f_rosenbrock(x0):.6f}")
print(f"||∇f(x₀)|| = {np.linalg.norm(df_rosenbrock(x0)):.6f}")
print(f"∇f(x₀) = {df_rosenbrock(x0)}")

# Verificar en el óptimo
print(f"\nVerificación en el óptimo teórico:")
print(f"f(x*) = {f_rosenbrock(x_star):.6f}")
print(f"||∇f(x*)|| = {np.linalg.norm(df_rosenbrock(x_star)):.2e}")


configs = {
    'aggressive': {
        'alpha': 0.01,
        'maxIter': 2000,
        'eps': 1e-5,
        'description': 'Configuración agresiva'
    },
    'conservative': {
        'alpha': 0.001,
        'maxIter': 5000,
        'eps': 1e-6,
        'description': 'Configuración conservadora'
    },
    'balanced': {
        'alpha': 0.005,
        'maxIter': 3000,
        'eps': 1e-5,
        'description': 'Configuración balanceada'
    }
}

# Probar con configuración balanceada
config = configs['balanced']

print(f"\nUsando configuración: {config['description']}")
print(f"α = {config['alpha']}, maxIter = {config['maxIter']}, ε = {config['eps']}")

np.random.seed(42)

methods = {}


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
    
    # Desempacar resultados
    if len(result) == 6:  
        best, val, xs, fs, errors, converged = result
        iters = len(xs) - 1
    elif len(result) == 7:  
        best, val, xs, fs, errors, converged, iters = result
    else:
        print(f"Formato de resultado inesperado: {len(result)} elementos")
        continue
    
    # Calcular métricas
    distance_to_opt = np.linalg.norm(best - x_star)
    value_error = abs(val - f_star)
    final_gradient_norm = np.linalg.norm(df_rosenbrock(best))
    
    print(f"  Solución encontrada: x = [{best[0]:.6f}, {best[1]:.6f}]")
    print(f"  Valor de función: f(x) = {val:.6f}")
    print(f"  Convergió: {'✓' if converged else '✗'}")
    print(f"  Iteraciones: {iters}")
    print(f"  ||∇f(x)||: {final_gradient_norm:.2e}")
    print(f"  Distancia al óptimo: ||x - x*|| = {distance_to_opt:.6f}")
    print(f"  Error en valor: |f(x) - f*| = {value_error:.6f}")
    
    # Evaluación específica para Rosenbrock
    if converged:
        if distance_to_opt < 0.01 and value_error < 0.01:
            performance = "EXCELENTE"
        elif distance_to_opt < 0.1 and value_error < 1.0:
            performance = "BUENO"
        elif distance_to_opt < 0.5 and value_error < 10.0:
            performance = "REGULAR"
        else:
            performance = "POBRE"
    else:
        # Para Rosenbrock, evaluar progreso incluso sin convergencia formal
        if distance_to_opt < 0.1 and value_error < 1.0:
            performance = "BUEN PROGRESO (sin convergencia formal) "
        elif distance_to_opt < 1.0 and value_error < 100.0:
            performance = "PROGRESO MODERADO "
        else:
            performance = "POCO PROGRESO "
    
    print(f"  Evaluación: {performance}")

# -------------------------------
# Tabla resumen
# -------------------------------
print(f"\n{'='*100}")
print("TABLA RESUMEN - FUNCIÓN ROSENBROCK")
print("="*100)
print(f"{'Método':<18} {'Conv':<5} {'Iter':<6} {'f(x)':<12} {'||x-x*||':<10} {'||∇f||':<10} {'Evaluación':<20}")
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
    final_gradient_norm = np.linalg.norm(df_rosenbrock(best))
    
    conv_str = "✓" if converged else "✗"
    
    # Evaluación más permisiva para Rosenbrock
    if distance_to_opt < 0.01:
        eval_str = "EXCELENTE"
    elif distance_to_opt < 0.1:
        eval_str = "BUENO"
    elif distance_to_opt < 0.5:
        eval_str = "REGULAR"
    elif distance_to_opt < 1.0:
        eval_str = "PROGRESO"
    else:
        eval_str = "POBRE"
    
    print(f"{name:<18} {conv_str:<5} {iters:<6d} {val:<12.6f} {distance_to_opt:<10.6f} "
          f"{final_gradient_norm:<10.2e} {eval_str:<20}")


# -------------------------------
# Gráfica de convergencia
# -------------------------------
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
plt.title("Comparación de convergencia en los 5 métodos (Rosenbrock 2D)")
plt.legend()
plt.grid(True)
plt.show()
