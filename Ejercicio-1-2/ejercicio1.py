import numpy as np

def random_direction(n):
    """Genera una dirección aleatoria unitaria"""
    d = np.random.randn(n)
    return d / np.linalg.norm(d)

def line_search_backtracking(f, df, x, d, alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=50):
    """
    Búsqueda de línea con backtracking (condición de Armijo)
    """
    alpha = alpha_init
    fx = f(x)
    gx = df(x)
    slope = np.dot(gx, d)
    
    for _ in range(max_iter):
        if f(x + alpha * d) <= fx + c1 * alpha * slope:
            return alpha
        alpha *= rho
    
    return alpha

def check_convergence(x_old, x_new, f_old, f_new, gradient, criterion, eps):
    """Verifica convergencia según el criterio especificado"""
    if criterion == 'gradient_norm':
        return np.linalg.norm(gradient) < eps
    elif criterion == 'function_change':
        return abs(f_new - f_old) < eps
    elif criterion == 'solution_change':
        return np.linalg.norm(x_new - x_old) < eps
    else:
        raise ValueError(f"Criterio desconocido: {criterion}")

def calculate_error(x_old, x_new, f_old, f_new, gradient, criterion):
    """Calcula el error según el criterio especificado"""
    if criterion == 'gradient_norm':
        return np.linalg.norm(gradient)
    elif criterion == 'function_change':
        return abs(f_new - f_old)
    elif criterion == 'solution_change':
        return np.linalg.norm(x_new - x_old)


# 1. Descenso gradiente naive con dirección aleatoria

def gd_random(f, df, x0, alpha=0.1, maxIter=1000, eps=1e-6, criterion='gradient_norm', 
              line_search=False):
    """
    Descenso gradiente con dirección aleatoria mejorado
    """
    x = np.array(x0, dtype=float)
    best, best_val = x.copy(), f(x)
    xs, fs, errors = [x.copy()], [best_val], []

    for _ in range(maxIter):
        g = df(x)
        f_current = f(x)
        
        # Verificar convergencia temprana por gradiente
        if np.linalg.norm(g) < eps and criterion == 'gradient_norm':
            return best, best_val, xs, fs, errors, True, len(xs)-1

        # Generar dirección aleatoria de descenso
        d = random_direction(len(x))
        if np.dot(g, d) > 0:
            d = -d

        # Determinar tamaño de paso
        step_size = line_search_backtracking(f, df, x, d) if line_search else alpha
        x_new = x + step_size * d
        f_new = f(x_new)

        # Calcular error y verificar convergencia
        error = calculate_error(x, x_new, f_current, f_new, g, criterion)
        
        xs.append(x_new.copy())
        fs.append(f_new)
        errors.append(error)

        if f_new < best_val:
            best, best_val = x_new.copy(), f_new
            
        if check_convergence(x, x_new, f_current, f_new, g, criterion, eps):
            return best, best_val, xs, fs, errors, True, len(xs)-1

        x = x_new

    return best, best_val, xs, fs, errors, False, maxIter


# 2. Descenso máximo naive

def gd_naive(f, df, x0, alpha=0.1, maxIter=1000, eps=1e-6, criterion='gradient_norm',
             line_search=False):
    """
    Descenso de gradiente clásico mejorado
    """
    x = np.array(x0, dtype=float)
    best, best_val = x.copy(), f(x)
    xs, fs, errors = [x.copy()], [best_val], []

    for _ in range(maxIter):
        g = df(x)
        f_current = f(x)
        
        if np.linalg.norm(g) < eps and criterion == 'gradient_norm':
            return best, best_val, xs, fs, errors, True, len(xs)-1

        d = -g  # Dirección de máximo descenso
        
        # Determinar tamaño de paso
        step_size = line_search_backtracking(f, df, x, d) if line_search else alpha
        x_new = x + step_size * d
        f_new = f(x_new)

        # Calcular error y verificar convergencia
        error = calculate_error(x, x_new, f_current, f_new, g, criterion)
        
        xs.append(x_new.copy())
        fs.append(f_new)
        errors.append(error)

        if f_new < best_val:
            best, best_val = x_new.copy(), f_new
            
        if check_convergence(x, x_new, f_current, f_new, g, criterion, eps):
            return best, best_val, xs, fs, errors, True, len(xs)-1

        x = x_new

    return best, best_val, xs, fs, errors, False, maxIter


# 3. Método de Newton

def newton(f, df, ddf, x0, alpha=1.0, maxIter=1000, eps=1e-6, criterion='gradient_norm',
           line_search=True, regularization=True):
    """
    Método de Newton con control de paso y regularización
    """
    x = np.array(x0, dtype=float)
    best, best_val = x.copy(), f(x)
    xs, fs, errors = [x.copy()], [best_val], []

    for _ in range(maxIter):
        g = df(x)
        H = ddf(x)
        f_current = f(x)
        
        if np.linalg.norm(g) < eps and criterion == 'gradient_norm':
            return best, best_val, xs, fs, errors, True, len(xs)-1

        try:
            # Regularización del Hessiano si es necesario
            if regularization:
                reg_param = 1e-6
                while True:
                    try:
                        H_reg = H + reg_param * np.eye(len(x))
                        d = -np.linalg.solve(H_reg, g)
                        break
                    except np.linalg.LinAlgError:
                        reg_param *= 10
                        if reg_param > 1e-2:
                            d = -g  # Fallback a gradiente descendente
                            break
            else:
                d = -np.linalg.solve(H, g)
                
        except np.linalg.LinAlgError:
            d = -g  # Fallback a gradiente descendente

        # Determinar tamaño de paso
        step_size = line_search_backtracking(f, df, x, d) if line_search else alpha
        x_new = x + step_size * d
        f_new = f(x_new)

        # Calcular error y verificar convergencia
        error = calculate_error(x, x_new, f_current, f_new, g, criterion)
        
        xs.append(x_new.copy())
        fs.append(f_new)
        errors.append(error)

        if f_new < best_val:
            best, best_val = x_new.copy(), f_new
            
        if check_convergence(x, x_new, f_current, f_new, g, criterion, eps):
            return best, best_val, xs, fs, errors, True, len(xs)-1

        x = x_new

    return best, best_val, xs, fs, errors, False, maxIter


# 4. Gradiente conjugado completo

def conjugate_gradient(f, df, x0, maxIter=1000, eps=1e-6, criterion='gradient_norm',
                      method='fletcher_reeves', line_search=True, restart_freq=None):
    """
    Gradiente conjugado con múltiples variantes y reinicio
    """
    x = np.array(x0, dtype=float)
    g = df(x)
    d = -g
    best, best_val = x.copy(), f(x)
    xs, fs, errors = [x.copy()], [best_val], []
    
    # Frecuencia de reinicio automática
    if restart_freq is None:
        restart_freq = len(x)

    for k in range(maxIter):
        f_current = f(x)
        
        if np.linalg.norm(g) < eps and criterion == 'gradient_norm':
            return best, best_val, xs, fs, errors, True, k

        # Determinar tamaño de paso
        alpha = line_search_backtracking(f, df, x, d) if line_search else 1e-2
        x_new = x + alpha * d
        f_new = f(x_new)
        g_new = df(x_new)

        # Calcular error y verificar convergencia
        error = calculate_error(x, x_new, f_current, f_new, g, criterion)
        
        xs.append(x_new.copy())
        fs.append(f_new)
        errors.append(error)

        if f_new < best_val:
            best, best_val = x_new.copy(), f_new
            
        if check_convergence(x, x_new, f_current, f_new, g, criterion, eps):
            return best, best_val, xs, fs, errors, True, k

        # Reinicio periódico o cuando es necesario
        if (k + 1) % restart_freq == 0 or np.dot(g_new, g) / np.dot(g, g) > 0.2:
            d = -g_new
        else:
            # Calcular β según el método seleccionado
            if method == 'fletcher_reeves':
                beta = np.dot(g_new, g_new) / np.dot(g, g)
            elif method == 'hestenes_stiefel':
                y = g_new - g
                beta = np.dot(g_new, y) / np.dot(d, y) if np.dot(d, y) != 0 else 0
            elif method == 'polak_ribiere':
                y = g_new - g
                beta = np.dot(g_new, y) / np.dot(g, g)
            else:
                raise ValueError(f"Método desconocido: {method}")
            
            # Asegurar β no negativo
            beta = max(0, beta)
            d = -g_new + beta * d

        x, g = x_new, g_new

    return best, best_val, xs, fs, errors, False, maxIter


# 5. Método BFGS mejorado

def bfgs(f, df, x0, maxIter=1000, eps=1e-6, criterion='gradient_norm', 
         line_search=True, reset_hessian=True):
    """
    BFGS mejorado con reinicio de Hessiano y búsqueda de línea
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)  # Aproximación inicial del Hessiano inverso
    g = df(x)
    best, best_val = x.copy(), f(x)
    xs, fs, errors = [x.copy()], [best_val], []

    for k in range(maxIter):
        f_current = f(x)
        
        if np.linalg.norm(g) < eps and criterion == 'gradient_norm':
            return best, best_val, xs, fs, errors, True, k

        d = -H @ g
        
        # Determinar tamaño de paso
        alpha = line_search_backtracking(f, df, x, d) if line_search else 1e-2
        x_new = x + alpha * d
        f_new = f(x_new)
        g_new = df(x_new)

        # Calcular error y verificar convergencia
        error = calculate_error(x, x_new, f_current, f_new, g, criterion)
        
        xs.append(x_new.copy())
        fs.append(f_new)
        errors.append(error)

        if f_new < best_val:
            best, best_val = x_new.copy(), f_new
            
        if check_convergence(x, x_new, f_current, f_new, g, criterion, eps):
            return best, best_val, xs, fs, errors, True, k

        # Actualización BFGS
        s = x_new - x
        y = g_new - g
        sy = np.dot(s, y)

        if abs(sy) > 1e-10:  # Verificar condición de curvatura
            rho = 1.0 / sy
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        elif reset_hessian and k % (2 * n) == 0:
            # Reiniciar Hessiano periódicamente
            H = np.eye(n)

        x, g = x_new, g_new

    return best, best_val, xs, fs, errors, False, maxIter

