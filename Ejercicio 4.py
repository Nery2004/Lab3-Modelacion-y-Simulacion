import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

"""
Problema: Ajustar el modelo
    f(x) = β0 + β1 x + β2 x² + β3 sin(7x) + β4 sin(13x)
minimizando la función
    E_λ(β) = Σ (f(x_i) - y_i)^2 + λ Σ (f(x_{i+1}) - f(x_i))^2.

Implementación: resolvemos en forma cerrada con álgebra lineal (sin SciPy).
"""

# Cargar los datos (x, y) desde 'datos_lab3.csv'.
# Se asume que el CSV está en la misma carpeta que este script y contiene columnas 'x' y 'y'.
csv_path = Path(__file__).resolve().parent / 'datos_lab3.csv'
data = pd.read_csv(csv_path)
x = data['x'].to_numpy(dtype=float)
y = data['y'].to_numpy(dtype=float)

# Ordenar por x para que el término de diferencias f(x_{i+1})-f(x_i) tenga sentido temporal
order = np.argsort(x)
x = x[order]
y = y[order]
n = x.size

# Definir la función del modelo
def model(x, beta):
    """
    f(x) = β0 + β1*x + β2*x² + β3*sin(7x) + β4*sin(13x)
    """
    return (beta[0] + beta[1]*x + beta[2]*x**2 + 
            beta[3]*np.sin(7*x) + beta[4]*np.sin(13*x))

def design_matrix(x: np.ndarray) -> np.ndarray:
    """Matriz de diseño Φ = [1, x, x², sin(7x), sin(13x)]."""
    return np.column_stack([
        np.ones_like(x),
        x,
        x**2,
        np.sin(7 * x),
        np.sin(13 * x),
    ])


def difference_matrix(n: int) -> np.ndarray:
    """Matriz D de primeras diferencias de tamaño (n-1) x n: (Dz)_i = z_{i+1} - z_i."""
    if n < 2:
        return np.zeros((0, n))
    D = np.zeros((n - 1, n))
    i = np.arange(n - 1)
    D[i, i] = -1.0
    D[i, i + 1] = 1.0
    return D

# Función para resolver el problema de optimización
def solve_regression(lambda_val: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resuelve en forma cerrada:
    min_β ||Φβ - y||² + λ ||D Φβ||²  =>  (ΦᵀΦ + λ Φᵀ Dᵀ D Φ) β = Φᵀ y
    """
    Phi = design_matrix(x)  # n x 5
    D = difference_matrix(x.size)  # (n-1) x n

    A = Phi.T @ Phi
    if lambda_val != 0 and D.size > 0:
        A = A + lambda_val * (Phi.T @ (D.T @ (D @ Phi)))
    b = Phi.T @ y

    # Resolver de forma estable (mínimos cuadrados por si A es mal condicionada)
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    return beta

# Resolver para los tres casos
print("Resolviendo para λ=0...")
beta_0 = solve_regression(0.0, x, y)

print("Resolviendo para λ=100...")
beta_100 = solve_regression(100.0, x, y)

print("Resolviendo para λ=500...")
beta_500 = solve_regression(500.0, x, y)

# Crear puntos para graficar las curvas (más densos para suavidad)
x_plot = np.linspace(min(x), max(x), 1000)
y_pred_0 = model(x_plot, beta_0)
y_pred_100 = model(x_plot, beta_100)
y_pred_500 = model(x_plot, beta_500)

# Graficar resultados
plt.figure(figsize=(12, 8))
plt.scatter(x, y, alpha=0.6, label='Datos originales', color='gray')
plt.plot(x_plot, y_pred_0, label=f'λ=0', linewidth=2, color='red')
plt.plot(x_plot, y_pred_100, label=f'λ=100', linewidth=2, color='blue')
plt.plot(x_plot, y_pred_500, label=f'λ=500', linewidth=2, color='green')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de modelos de regresión con diferentes valores de λ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Mostrar coeficientes obtenidos
print("\nCoeficientes obtenidos:")
print(f"λ=0:   {beta_0}")
print(f"λ=100: {beta_100}")
print(f"λ=500: {beta_500}")

# Calcular errores de ajuste para comparación
error_fit_0 = np.sum((model(x, beta_0) - y)**2)
error_fit_100 = np.sum((model(x, beta_100) - y)**2)
error_fit_500 = np.sum((model(x, beta_500) - y)**2)

print(f"\nError de ajuste (sin regularización):")
print(f"λ=0:   {error_fit_0:.4f}")
print(f"λ=100: {error_fit_100:.4f}")
print(f"λ=500: {error_fit_500:.4f}")