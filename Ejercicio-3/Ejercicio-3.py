# Universidad del Valle de Guatemala
# Modelacion y Simulación
# Laboratorio 3 - Ejercicio 3

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def constSumaGauss(k=8, sigma=0.8, semilla=42):
    generador = np.random.default_rng(semilla)
    centros = generador.uniform(0, 8, size=(k, 2))

    def funcion(x):
        x = np.array(x)
        valores = np.sum(np.exp(-np.linalg.norm(x - centros, axis=1) ** 2 / (2 * sigma)))
        return -valores

    def gradiente(x):
        x = np.array(x)
        diferencias = x - centros
        pesos = np.exp(-np.linalg.norm(diferencias, axis=1) ** 2 / (2 * sigma))
        g = np.sum((diferencias / sigma) * pesos[:, None], axis=0)
        return -g

    def hessiano(x):
        x = np.array(x)
        diferencias = x - centros
        pesos = np.exp(-np.linalg.norm(diferencias, axis=1) ** 2 / (2 * sigma))
        H = np.zeros((2, 2))
        for d, w in zip(diferencias, pesos):
            termino1 = np.eye(2) * (w / sigma)
            termino2 = (np.outer(d, d) / (sigma ** 2)) * w
            H += (termino1 - termino2)
        return -H

    return funcion, gradiente, hessiano, centros

def armijoRetroceso(funcion, gradiente, x, direccion, alfa0=1.0, beta=0.5, c=1e-4, maxIter=50):
    alfa = alfa0
    fx = funcion(x)
    gx = gradiente(x)
    gDotD = np.dot(gx, direccion)
    
    if gDotD >= 0:
        return 1e-6
    
    for _ in range(maxIter):
        if funcion(x + alfa * direccion) <= fx + c * alfa * gDotD:
            return alfa
        alfa *= beta
    return alfa

def bfgsArmijo(funcion, gradiente, x0, maxIter=500, tol=1e-8):
    x = np.array(x0, dtype=float)
    n = len(x)
    BInversa = np.eye(n)
    tray = [x.copy()]
    
    for iteracion in range(maxIter):
        g = gradiente(x)
        if np.linalg.norm(g) < tol:
            break
        
        direccion = -BInversa.dot(g)
        alfa = armijoRetroceso(funcion, gradiente, x, direccion, alfa0=1.0)
        s = alfa * direccion
        xNuevo = x + s
        gNuevo = gradiente(xNuevo)
        y = gNuevo - g
        ys = np.dot(y, s)
        
        if ys > 1e-12:
            sCol = s.reshape(-1,1)
            yCol = y.reshape(-1,1)
            rho = 1.0 / float(yCol.T.dot(sCol))
            I = np.eye(n)
            V = I - rho * sCol.dot(yCol.T)
            BInversa = V.dot(BInversa).dot(V.T) + rho * sCol.dot(sCol.T)
        else:
            BInversa = np.eye(n)
        
        x = xNuevo
        tray.append(x.copy())
    
    return x, tray

def agruparSoluciones(soluciones, funcion, tol=0.08):
    grupos = []
    for sol in soluciones:
        colocado = False
        for grupo in grupos:
            if np.linalg.norm(sol - grupo["representante"]) < tol:
                grupo["puntos"].append(sol)
                grupo["representante"] = np.mean(grupo["puntos"], axis=0)
                colocado = True
                break
        if not colocado:
            grupos.append({"representante": sol.copy(), "puntos": [sol]})
    resultado = []
    for grupo in grupos:
        representante = grupo["representante"]
        resultado.append((representante, funcion(representante), len(grupo["puntos"])))
    return resultado

def principal():
    funcion, gradiente, hessiano, centros = constSumaGauss(k=8, sigma=0.8, semilla=42)
    os.makedirs("gaussResults", exist_ok=True)

    X, Y = np.meshgrid(np.linspace(0, 8, 200), np.linspace(0, 8, 200))
    Z = np.array([[funcion([x, y]) for x in X[0]] for y in Y[:, 0]])
    plt.figure(figsize=(6, 5))
    CS = plt.contour(X, Y, Z, 30)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.scatter(centros[:, 0], centros[:, 1], c='blue', marker='x', label="centros")
    plt.legend()
    plt.title("Contorno de f y centros (x_i)")
    plt.savefig("gaussResults/global.png", dpi=150)
    plt.close()

    generador = np.random.default_rng(0)
    inicios = generador.uniform(0, 8, size=(50, 2))
    soluciones = []
    trayectorias = []
    for x0 in inicios:
        solucion, tray = bfgsArmijo(funcion, gradiente, x0, maxIter=500, tol=1e-8)
        soluciones.append(solucion)
        trayectorias.append(tray)

    grupos = agruparSoluciones(soluciones, funcion, tol=0.12)

    df = pd.DataFrame([{
        "x1": representante[0], "x2": representante[1], "f(x)": valor, "vecesEncontrado": cantidad
    } for representante, valor, cantidad in grupos])
    df = df.sort_values("f(x)")
    df.to_csv("gaussResults/tablaMin.csv", index=False)

    print("Centros (x_i):")
    print(centros)
    print("\nTabla de mínimos:")
    print(df)

    for idx, (representante, valor, cantidad) in enumerate(grupos, 1):
        plt.figure(figsize=(6, 5))
        plt.contour(X, Y, Z, 30)
        for tray in trayectorias:
            if np.linalg.norm(tray[-1] - representante) < 0.12:
                tray = np.array(tray)
                plt.plot(tray[:, 0], tray[:, 1], marker=".", markersize=3)
        plt.title(f"Mínimo {idx}: f={valor:.6f}, encontrado {cantidad} veces")
        plt.savefig(f"gaussResults/min{idx}trayectorias.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    principal()
