# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:11:34 2025

@author: Ricar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Complejos_Simpliciales import SimplicialCNN
from Complejos_Simpliciales import scipy_to_torch_sparse
import dataloader
import utility
import scipy.sparse as sp
import earlystopping
import optuna
from optuna.trial import Trial
from scipy.sparse.linalg import eigsh
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Configuración básica
N = 2
total_iters = 500
gso_type = 'sym_norm_lap'

torch.manual_seed(1337)
np.random.seed(1337)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 1. FUNCIÓN OBJETIVO PARA OPTUNA
# -----------------------
def objective(trial: Trial):
    # SUGERIR HIPERPARÁMETROS
    hidden_features = trial.suggest_categorical('hidden_features', [8, 16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)  
    N = trial.suggest_int('N', 2, 3)  # Ajustar 
    
    # Cargar datos (mismo que antes)
    feature, gso, label, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device)
    
    # Crear modelo con hiperparámetros sugeridos
    network = SimplicialCNN(in_features=n_feat,
                           hidden_features=hidden_features, 
                           num_classes=n_class,
                           N=N)
    network.to(device)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    
    # Entrenamiento abreviado para búsqueda
    best_val_acc = 0
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(100):  # Menos épocas para búsqueda rápida
        optimizer.zero_grad()
        output = network(feature, gso)
        loss_train = loss_fn(output[idx_train], label[idx_train])
        loss_train.backward()
        optimizer.step()
        
        # Validación
        network.eval()
        with torch.no_grad():
            output_val = network(feature, gso)
            acc_val = utility.calc_accuracy(output_val[idx_val], label[idx_val])
            acc_val = acc_val.item()
        
        # Early stopping simple
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            break
    
    return best_val_acc

# -----------------------
# 2. ESTUDIO DE OPTUNA
# -----------------------
def run_optuna_study():
    study = optuna.create_study(
        direction='maximize',  # Maximizar accuracy
        sampler=optuna.samplers.TPESampler(seed=1337),
        pruner=optuna.pruners.HyperbandPruner()  # Podar trials poco prometedores
    )
    
    study.optimize(
        objective,
        n_trials=100,  # Número de combinaciones a probar
        n_jobs=1,     # Paralelizar
        show_progress_bar=True
    )
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    print("="*50)
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")
    print(f"Mejor accuracy de validación: {study.best_value:.4f}")
    
    return study

# -----------------------
# 3. ENTRENAMIENTO FINAL CON MEJORES PARÁMETROS
# -----------------------
def train_final_model(best_params):
    feature, gso, label, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device)
    
    network = SimplicialCNN(
        in_features=n_feat,
        hidden_features=best_params['hidden_features'], 
        num_classes=n_class,
        N=best_params['N']
    )
    network.to(device)
    
    optimizer = torch.optim.Adam(
        network.parameters(), 
        lr=best_params['learning_rate'], 
        weight_decay=best_params['weight_decay']
    )
    loss_fn = nn.NLLLoss()
    early_stopping = earlystopping.EarlyStopping(patience=50, verbose=True)
    

    train_accs = []
    val_accs = []
    losses = []
    
    for i in range(total_iters):
        optimizer.zero_grad()
        output = network(feature, gso)
        loss_train = loss_fn(output[idx_train], label[idx_train])
        acc_train = utility.calc_accuracy(output[idx_train], label[idx_train])
        loss_train.backward()
        optimizer.step()
        
        # Validación
        loss_val, acc_val = val(network, label, output, loss_fn, idx_val)
        
        print('Epoch: {:03d} | LR: {:.8f} | Train loss: {:.6f} | Train acc: {:.6f} | Val loss: {:.6f} | Val acc: {:.6f}'.format(
            i+1, optimizer.param_groups[0]['lr'], loss_train.item(), acc_train.item(), 
            loss_val.item(), acc_val.item()))
        
        losses.append(loss_train.item())
        train_accs.append(acc_train.item())
        val_accs.append(acc_val.item())
        
        early_stopping(loss_val, network)
        if early_stopping.early_stop:
            network.load_state_dict(torch.load("./chebynet.pth"))
            print('Early stopping.')
            break
    
    # Evaluación final
    with torch.no_grad():
        output_final = network(feature, gso)
        pred_final = output_final.argmax(dim=1)
        test_acc = (pred_final[idx_test] == label[idx_test]).float().mean()
        
        print(f"\n🎯 RESULTADOS FINALES:")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Train Accuracy: {(pred_final[idx_train] == label[idx_train]).float().mean():.4f}")
        print(f"Val Accuracy: {(pred_final[idx_val] == label[idx_val]).float().mean():.4f}")
    
    return network, test_acc.item()

# -----------------------
# 4. FUNCIONES AUXILIARES (las que ya tenías)
# -----------------------
def process_data(device):
    
    feature, L_1, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_data()
    
        
    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    
    #Aqui se va a normalizar
    def escalar_laplaciano_por_espectro(L):
        # Calcular el mayor autovalor (k=1, largest=True)
        lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        #L_scaled = (2.0 / lambda_max) * L - sp.eye(L.shape[0])
        L_scaled = L / (lambda_max + 1e-8)  # [0, 1]
        return L_scaled
   
    gso = escalar_laplaciano_por_espectro(L_1)
    gso = utility.calc_chebynet_gso(gso)

    if sp.issparse(feature):
        feature = utility.cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = torch.from_numpy(feature).to(device)
    gso = utility.cnv_sparse_mat_to_coo_tensor(gso, device)
    label = torch.LongTensor(label).to(device)

    return feature, gso, label, idx_train, idx_val, idx_test, n_feat, n_class

def val(model, label, output, loss, idx_val):
    model.eval()
    with torch.no_grad():
        loss_val = loss(output[idx_val], label[idx_val])
        acc_val = utility.calc_accuracy(output[idx_val], label[idx_val])
    return loss_val, acc_val

# -----------------------
# 5. EJECUCIÓN PRINCIPAL
# -----------------------
if __name__ == "__main__":
    # Ejecutar búsqueda de hiperparámetros
    print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON OPTUNA...")
    study = run_optuna_study()
    
    # Entrenar modelo final con mejores parámetros
    print("\nENTRENANDO MODELO FINAL CON MEJORES HIPERPARÁMETROS...")
    best_model, final_test_acc = train_final_model(study.best_params)
    
    # Guardar resultados
    np.savez_compressed(
        "resultados_optuna.npz",
        best_params=study.best_params,
        best_val_accuracy=study.best_value,
        final_test_accuracy=final_test_acc,
        all_trials=study.trials_dataframe()
    )
    
   