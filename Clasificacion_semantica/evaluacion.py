# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 11:29:10 2025

@author: Ricar
"""

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from Complejos_Simpliciales import SimplicialCNN
import dataloader
import utility
from con_optuna import process_data
import pandas as pd


def cargar_modelo(path, n_feat, hidden_features, n_class, N, device):
    model = SimplicialCNN(
        in_features=n_feat,
        hidden_features=hidden_features,
        num_classes=n_class,
        N=N
    )
    
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


def evaluar_modelo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feature, gso, labels, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device)
    feature = torch.FloatTensor(feature).to(device)
    
    labels = torch.LongTensor(labels).to(device)

    # Cargar mejores hiperparámetros
    data = np.load("resultados_optuna.npz", allow_pickle=True)
    best = data["best_params"].item()

    hidden_features = best["hidden_features"]
    N = best["N"]

    # Cargar modelo entrenado
    BEST_MODEL_PATH = "chebynet.pth"
    model = cargar_modelo(BEST_MODEL_PATH, n_feat, hidden_features, n_class, N, device)
    
    
    # Cargar datos originales para mapear índices a palabras
    df_features = pd.read_csv("features_aristas.csv")
    
    
    # Predicciones
    with torch.no_grad():
        output = model(feature, gso)
        pred = output.argmax(dim=1)

    y_true = labels[idx_test].cpu().numpy()
    y_pred = pred[idx_test].cpu().numpy()

    print("\n=== ACCURACY EN TEST ===")
    print((y_pred == y_true).mean())

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cognado", "Falso", "Semantico"],
                yticklabels=["Cognado", "Falso", "Semantico"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.title("Matriz de Confusión Simplicial")
    plt.tight_layout()
    plt.show()
    
    
    # IDENTIFICAR PAIRES MAL CLASIFICADOS
    print("\n=== PAIRES MAL CLASIFICADOS ===")
    print("="*60)
    
    errores = []
    
    for i, test_idx in enumerate(idx_test.cpu().numpy()):
        true_label = y_true[i]
        pred_label = y_pred[i]
        
        if true_label != pred_label:
            # Obtener el par del dataframe
            if test_idx < len(df_features):
                fila = df_features.iloc[test_idx]
                
                error_info = {
                    'indice': test_idx,
                    'palabra1': fila['palabra1'],
                    'palabra2': fila['palabra2'],
                    'idioma1': fila['idioma1'],
                    'idioma2': fila['idioma2'],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'dist_coseno': fila.get('dist_coseno', None),
                    'sim_ort': fila.get('sim_ort', None)
                }
                errores.append(error_info)
    
    # Mostrar errores
    if errores:
        print(f"\nTotal de errores: {len(errores)}")
        print("\nLista de pares mal clasificados:")
        print("-"*80)
        
        for i, error in enumerate(errores[:20]):  # Mostrar primeros 20
            # Mapear etiquetas
            label_map = {0: "Cognado", 1: "Falso", 2: "Otro"}
            true_str = label_map.get(error['true_label'], f"Clase {error['true_label']}")
            pred_str = label_map.get(error['pred_label'], f"Clase {error['pred_label']}")
            
            print(f"{i+1:3d}. {error['palabra1']:15s} ({error['idioma1']}) - "
                  f"{error['palabra2']:15s} ({error['idioma2']})")
            print(f"     Real: {true_str:10s} | Pred: {pred_str:10s} | "
                  f"DistCos: {error['dist_coseno']:.3f} | "
                  f"SimOrt: {error['sim_ort']:.3f}")
            print()

if __name__ == "__main__":
    evaluar_modelo()
