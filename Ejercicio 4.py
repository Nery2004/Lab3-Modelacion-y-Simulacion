import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Cargar los datos
# Asumiendo que el archivo datos.lab3.csv tiene dos columnas: x y y
data = pd.read_csv('datos.lab3.csv')
x = data['x'].values
y = data['y'].values
n = len(x)

# Definir la función del modelo
def model(x, beta):
    """
    f(x) = β0 + β1*x + β2*x² + β3*sin(7x) + β4*sin(13x)
    """
    return (beta[0] + beta[1]*x + beta[2]*x**2 + 
            beta[3]*np.sin(7*x) + beta[4]*np.sin(13*x))

# Definir la función de error regularizada Eλ(β)
def error_function(beta, lambda_val, x, y):
    """
    Eλ(β) = Σ(y_pred - y)^2 + λ*Σ(f(x_{i+1}) - f(x_i))^2
    """
    # Calcular predicciones
    y_pred = model(x, beta)
    
    # Error de ajuste (primer término)
    error_fit = np.sum((y_pred - y)**2)
    
    # Error de regularización (segundo término)
    # Calcular diferencias entre predicciones consecutivas
    diff_pred = np.diff(y_pred)
    error_reg = np.sum(diff_pred**2)
    
    # Error total
    total_error = error_fit + lambda_val * error_reg
    
    return total_error

# Función para resolver el problema de optimización
def solve_regression(lambda_val, x, y, initial_guess=None):
    """
    Resuelve el problema de optimización para un valor dado de λ
    """
    if initial_guess is None:
        initial_guess = np.zeros(5)  # β0, β1, β2, β3, β4
    
    # Función objetivo para el optimizador
    def objective(beta):
        return error_function(beta, lambda_val, x, y)
    
    # Optimizar usando método BFGS
    result = minimize(objective, initial_guess, method='BFGS')
    
    if result.success:
        return result.x
    else:
        raise ValueError(f"Optimización falló para λ={lambda_val}: {result.message}")

# Resolver para los tres casos
print("Resolviendo para λ=0...")
beta_0 = solve_regression(0, x, y)

print("Resolviendo para λ=100...")
beta_100 = solve_regression(100, x, y, beta_0)

print("Resolviendo para λ=500...")
beta_500 = solve_regression(500, x, y, beta_100)

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