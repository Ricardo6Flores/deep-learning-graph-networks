# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:57:21 2025

@author: Ricar
"""

"""
Analiza resultados de un entrenamiento simplicial: calcula métricas y grafica la mejor iteración.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def analizar_resultados(file_path, mostrar_grafica=True, N=2, hidden_features=2,
                        w_damaged=5, w_lineal=0.1, w_slope=0.1, w_intercept=0.05,
                         lambda_decay=0.5, seed=42, guardar = None, learning_rate=None):
    """
    Analiza un archivo .npz generado por run_experiment() y calcula:
    1) Score por iteración (dañados + regresión lineal)
    2) Score temporal ponderado exponencial sobre las últimas iteraciones
    """
    data = np.load(file_path, allow_pickle=True)
    mask_damaged_0 = np.load('mask_damaged_0.npy', allow_pickle=True)
    mask_known_0 = np.load('mask_known_0.npy', allow_pickle=True)

    outputs = data["outputs"].tolist()
    cochain_target = data["cochain_target"].tolist()[0]
    num_params = data["num_params"]
    data.close()
    iters = len(outputs)
    
    # --- listas para guardar valores crudos ---
    score_damaged_list = []
    score_lineal_list = []
    slope_err_list = []
    intercept_err_list = []
    
    score_list = [] #scores antes del temporal, pero despues de ya ponderados
    acc_damaged_list = []  #Solo distancias
    slope_list = []
    intercept_list = []

    for i in range(iters):
        pred = np.array(outputs[i])
        true = np.array(cochain_target[0])
        if len(pred) == 0:
            score_list.append(0)
            acc_damaged_list.append(0)
            slope_list.append(0)
            intercept_list.append(0)
            continue
        
        # --- Distancia de los dañados a y=x ---
        diff_damaged = np.abs(pred[mask_damaged_0] - true[mask_damaged_0])
        score_damaged = 1 / (1 + diff_damaged.sum())  # menor distancia -> mayor score
        score_damaged_list.append(score_damaged)
        acc_damaged_list.append(100 * score_damaged)
        
        # --- Ajuste lineal global ---
        model = LinearRegression()
        model.fit(true.reshape(-1,1), pred)
        slope = model.coef_[0]
        intercept = model.intercept_
        slope_list.append(slope)
        intercept_list.append(intercept)
        
        # Penalización slope/intercept y=x
        score_lineal = (1 - abs(slope-1)) * (1 - min(abs(intercept)/true.max(),1))
        score_lineal_list.append(score_lineal)
        slope_err_list.append(abs(slope-1))
        intercept_err_list.append(abs(intercept))

    # --- Normalización de 0 a 1 ---
    def normalize(x):
        x = np.array(x)
        return (x - x.min()) / (x.max() - x.min() + 1e-8)   
    
    score_damaged_list = np.array(score_damaged_list)
    score_lineal_norm = normalize(score_lineal_list)
    slope_norm = normalize(slope_err_list)       
    intercept_norm = normalize(intercept_err_list)

    # --- Score combinado ---
    score_list = (
        w_damaged * score_damaged_list - w_slope * slope_norm -
        w_intercept * intercept_norm
    )
    
    # --- Score temporal ponderado exponencial ---
    # k = max(1, int(iters * ventana_frac))
    # last_window = score_list[-k:]
    weights = np.exp(-lambda_decay * np.arange(iters-1, -1, -1))
    score_temporal = np.sum(np.array(score_list) * weights) / np.sum(weights)

    # --- Mejor iteración individual ---
    best_iter = int(np.nanargmax(score_list))
    best_score_iter = score_list[best_iter]
    mejor_iter =  int(np.nanargmax(acc_damaged_list))
    best_acc_damaged = acc_damaged_list[mejor_iter]
    best_slope = slope_list[best_iter]
    best_intercept = intercept_list[best_iter]

    print(f"\n Mejor iteración = {best_iter}")
    print(f"Score iteración = {best_score_iter:.3f}, Score temporal = {score_temporal:.3f}")
    print(f"Acc dañados = {best_acc_damaged:.2f}% | Recta: y = {best_slope:.3f}x + {best_intercept:.3f}")

    # --- Gráfica ---
    if mostrar_grafica:
        all_out = np.array(outputs[best_iter])
        plt.figure(figsize=(6,6))
        plt.scatter(true[mask_known_0], all_out[mask_known_0], alpha=0.6, s=20, color='blue', label='Conocidos')
        plt.scatter(true[mask_damaged_0], all_out[mask_damaged_0], alpha=0.8, s=20, color='red', label='Dañados')

        lims = [min(true.min(), all_out.min()), max(true.max(), all_out.max())]
        x_line = np.linspace(*lims, 100)
        plt.plot(x_line, best_slope*x_line + best_intercept, 'g-', label=f'Regresión')
        plt.plot(lims, lims, 'k--', label='Ideal')

        textstr = '\n'.join((
            f'Iter = {best_iter}',
            f'Score iter = {best_score_iter:.3f}',
            f'Score temporal = {score_temporal:.3f}',
            f'Acc dañados = {best_acc_damaged:.2f}%',
            f'y={best_slope:.2f}x+{best_intercept:.2f}',
            f'N={N}, filtros={hidden_features}',
            f'Learning_rate={learning_rate}'
        ))
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.xlabel("Valores reales")
        plt.ylabel("Predicciones")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"heatmaps/{guardar}/seed{seed}/mejor_iteracion_N{N}_h{hidden_features}_lr{learning_rate}_seed{seed}.png", dpi=300, bbox_inches='tight')
        plt.close()

        #plt.show()

    return best_iter, best_acc_damaged, score_temporal, num_params
