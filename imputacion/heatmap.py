# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:50:48 2025

@author: Ricar
"""

from entrenar_scnn import preparar_datos, run_experiment
from evaluacion import analizar_resultados
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from grafo_malla import grafo_malla
import csv

# # -------------------------------------------------------------
# # Parámetros iniciales del grafo
# # -------------------------------------------------------------

n = 20
m = 50
capa = 2
connectedness = 0.2
num_danados = 5
grad_max = 4

    
carpeta = f"grafo_conexo{n}vertices{m}aristas_{capa}capas_{connectedness}con_{num_danados}dañados_gradomax{grad_max}"
for seed in [41,42,43]:
    grafo = f"grafo_{n}vertices{m}aristas{num_danados}danados_seed{seed}"
    # -------------------------------------------------------------
    #  1️⃣ Generar grafo y archivos base
    # -------------------------------------------------------------
    print(f"Generando grafo {grafo}...")
    os.makedirs(f"heatmaps/{carpeta}/seed{seed}", exist_ok=True)
    grafo_malla(n=n, m=m, connectedness=connectedness, num_danados=num_danados, seed=seed, guardar=carpeta, grado = grad_max)
    print("Archivos del grafo generados.\n")
    
    
    # # -------------------------------------------------------------
    # #  Limpiar la carpeta de resultados antes de comenzar
    # # -------------------------------------------------------------
    resultados_dir = "Resultados"
    
    # Si la carpeta existe, eliminarla completamente
    if os.path.exists(resultados_dir):
        print(f" Eliminando carpeta antigua '{resultados_dir}'...")
        shutil.rmtree(resultados_dir)
    
    # Volver a crear la carpeta vacía
    os.makedirs(resultados_dir, exist_ok=True)
    print(f" Carpeta '{resultados_dir}' lista.\n")
    
    # -------------------------------------------------------------
    # 1️⃣ Cargar una sola vez los datos base
    # -------------------------------------------------------------
    data = preparar_datos()

    # -------------------------------------------------------------
    # 2️⃣ Definir los rangos de hiperparámetros
    # -------------------------------------------------------------
    Ns = [2, 3, 4, 5]  # Filas 
    Hiddens = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Columnas 
    
    #learning rates (filas x columnas)
    learning_rates_table = [
        ["x", "x", "x", "x", "x", 0.0001, "x", 0.00005, "x", 0.00003],
        [0.00022, 0.0005, "x", "x", "x", 0.00005, 0.00006, "x", "x", "x"],
        [None, None, 0.0001, None, None, None, None, None, None, None],
        [None, None, None, None, None, None, None, None, None, None]
    ]
    
    # Inicializar heatmaps con NaN (valores nulos)
    heatmap_acc = np.full((len(Ns), len(Hiddens)), np.nan)
    heatmap_r2 = np.full_like(heatmap_acc, np.nan)
    
    # -------------------------------------------------------------
    # 3️⃣ Barrido de combinaciones
    # -------------------------------------------------------------
    resultados = []
    for i, N in enumerate(Ns):
        for j, h in enumerate(Hiddens):
            lr = learning_rates_table[i][j]
            
            # Saltar si es "x", None o está vacío
            if lr in ["x", None, ""] or (isinstance(lr, (int, float)) and lr <= 0):
                print(f"  Saltando N={N}, h={h} (lr={lr})")
                continue
            
            save_prefix = f"resultados"
            
            print(f"Ejecutando N={N}, hidden={h}, lr={lr}")
    
            # Entrenar modelo
            best_loss = run_experiment(
                data,
                N=N,
                hidden_features=h,
                total_iters=2000,
                learning_rate=lr,
                save_prefix=save_prefix
            )
    
            # Analizar resultados y obtener métricas
            file_path = f"Resultados/{save_prefix}_N{N}_hidden{h}.npz"
            _, best_acc, best_score_temp, num_params = analizar_resultados(
                file_path, 
                mostrar_grafica=True, 
                N=N, 
                hidden_features=h, 
                seed=seed, 
                guardar=carpeta, 
                learning_rate=lr
            )
    
            # Llenar solo las celdas donde se ejecutó el experimento
            heatmap_acc[i, j] = best_acc
            heatmap_r2[i, j] = best_score_temp
            
            print(f" N={N}, hidden={h}: Accuracy={best_acc:.2f}%, score={best_score_temp:.3f}")
            
            resultados.append({
                "N": N,
                "hidden": h,
                "accuracy": best_acc,
                "score": best_score_temp,
                "params": num_params            
            })
    
    print("Barrido completado")
    # -------------------------------------------------------------
    # Heatmaps con valores anotados
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Título general de la figura
    fig.suptitle(grafo, fontsize=14, fontweight='bold')
    
    # --- Heatmap de Accuracy ---
    im1 = axes[0].imshow(heatmap_acc, cmap='viridis', origin='lower')
    axes[0].set_title("Puntaje máximo de iteracion") # de 1/(1+distancia_dañados)
    axes[0].set_xticks(range(len(Hiddens)))
    axes[0].set_yticks(range(len(Ns)))
    axes[0].set_xticklabels(Hiddens)
    axes[0].set_yticklabels(Ns)
    axes[0].set_xlabel("Filtros")
    axes[0].set_ylabel("N polinomio")
    plt.colorbar(im1, ax=axes[0])
    
    # Anotar valores numéricos
    for i in range(len(Ns)):
        for j in range(len(Hiddens)):
            axes[0].text(j, i, f"{heatmap_acc[i,j]:.1f}", ha='center', va='center', color='white', fontsize=10)
    
    # --- Heatmap del Score ---
    im2 = axes[1].imshow(heatmap_r2, cmap='plasma', origin='lower')
    axes[1].set_title("Score temporal")
    axes[1].set_xticks(range(len(Hiddens)))
    axes[1].set_yticks(range(len(Ns)))
    axes[1].set_xticklabels(Hiddens)
    axes[1].set_yticklabels(Ns)
    axes[1].set_xlabel("Filtros")
    axes[1].set_ylabel("N polinomio")
    plt.colorbar(im2, ax=axes[1])
    # Anotar valores del score
    for i in range(len(Ns)):
        for j in range(len(Hiddens)):
            axes[1].text(j, i, f"{heatmap_r2[i,j]:.1f}", ha='center', va='center', color='white', fontsize=10)
    
    
    
    # Ordenar por el score (de mayor a menor)
    resultados_ordenados = sorted(resultados, key=lambda x: x["score"], reverse=True)
    
    # --- Lista de top 5 hiperparámetros al lado derecho ---
    top5 = resultados_ordenados[:5]  # los 5 primeros
    top5_text = "2 Capas, con misma no filt \n Top 5 configuraciones:\n"
    for idx, r in enumerate(top5, 1):
        top5_text += f"{idx}. N={r['N']}, h={r['hidden']}, params={r['params']} | S={r['score']:.3f}\n"
    
    # Posicionar el texto a la derecha del heatmap derecho
    fig.text(0.95, 0.5, top5_text, fontsize=10, va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
    
    plt.tight_layout(rect=[0,0,0.92,1])  # dejar espacio para el texto a la derecha
    # Guardar la figura
    fig.savefig(f"heatmaps/{carpeta}/seed{seed}/"+grafo+f"_{capa}capa.png", dpi=300, bbox_inches='tight')
    print(f"Heatmap guardado: heatmaps/{grafo}.png\n")
    plt.close(fig)
    

    # Archivo CSV de resultados acumulativos
    csv_file = f"heatmaps/{carpeta}/resultados_acumulados.csv"
    
    # Revisar si existe; si no, escribir cabecera
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["grafo", "seed", "top5_str"])
    
    # Construir string de top5
    top5_str = "; ".join([f"N={r['N']},h={r['hidden']},params={r['params']},score={r['score']:.3f}" for r in top5])
    
    # Guardar en una sola fila
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([grafo, seed, top5_str])

