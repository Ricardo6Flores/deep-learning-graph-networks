# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 15:40:07 2025

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
# #############corar###########
# hidden_features: 32
# learning_rate: 0.004869811015441527
# weight_decay: 2.4535201815832375e-05
# N: 3
# ##############################

# ##### texas #############
# hidden_features: 64
# learning_rate: 0.0042251769794001744
# weight_decay: 1.1440805348957907e-05
# N: 2

#hidden_features: 64,
#learning_rate: 0.009653840842376289
#weight_decay: 6.26611128266062e-05
#N: 2
# ####################
# Configuración
#N = 2
N = 3
#hidden_features = 64
hidden_features = 32
#learning_rate = 0.001
learning_rate = 0.004869811015441527
total_iters = 500
gso_type = 'sym_norm_lap'  

torch.manual_seed(1337)
np.random.seed(1337)

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 1. CARGAR DATOS CORRECTAMENTE
# -----------------------
def process_data(device, gso_type):
    dataset = 'corar'
    #feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_webkb_data(dataset)
    feature, adj, label, idx_train, idx_val, idx_test, n_feat, n_class = dataloader.load_citation_data(dataset)

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)

    gso = utility.calc_gso(adj, gso_type)
    gso = utility.calc_chebynet_gso(gso)

    # Convertir a tensores
    if sp.issparse(feature):
        feature = utility.cnv_sparse_mat_to_coo_tensor(feature, device)
    else:
        feature = torch.from_numpy(feature).to(device)
    gso = utility.cnv_sparse_mat_to_coo_tensor(gso, device)
    label = torch.LongTensor(label).to(device)

    return feature, gso, label, idx_train, idx_val, idx_test, n_feat, n_class

# Cargar datos
feature, gso, label, idx_train, idx_val, idx_test, n_feat, n_class = process_data(device, gso_type)

# -----------------------
# 2. PREPARAR DATOS PARA 
# -----------------------
# Convertir características a formato [n_features, n_nodos]
x_input = feature # [2680, 302] para Cora
labels = label  # [2680]

# Crear máscaras booleanas 
train_mask = torch.zeros(len(labels), dtype=torch.bool)
train_mask[idx_train] = True
val_mask = torch.zeros(len(labels), dtype=torch.bool)  
val_mask[idx_val] = True
test_mask = torch.zeros(len(labels), dtype=torch.bool)
test_mask[idx_test] = True

# -----------------------
# 3. CREAR MODELO
# -----------------------
network = SimplicialCNN(in_features=n_feat,  # 302 para Cora
                       hidden_features=hidden_features, 
                       num_classes=n_class,  # 7 para Cora
                       N=N)
network.to(device)
#weight_decay = 0.0001
weight_decay = 2.4535201815832375e-05
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss = nn.NLLLoss()
patience = 50
early_stopping = earlystopping.EarlyStopping(patience=patience, verbose=True)

# Contar parámetros
num_params = sum(p.numel() for p in network.parameters())
print(f"Número total de parámetros: {num_params}")
print(f"Clases: {n_class}, Características: {n_feat}")
print(f"Nodos - Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

def val(model, label, output, loss, idx_val):
    model.eval()
    with torch.no_grad():
        loss_val = loss(output[idx_val], label[idx_val])
        acc_val = utility.calc_accuracy(output[idx_val], label[idx_val])

    return loss_val, acc_val

# -----------------------
# 4. BUCLE DE ENTRENAMIENTIO 
# -----------------------
train_accs = []
val_accs = []
losses = []

for i in range(total_iters):
    optimizer.zero_grad()
    
   
    output = network(x_input, gso)  # output: [n_nodos, n_class]
    
    loss_train = loss(output[idx_train], label[idx_train])
    acc_train = utility.calc_accuracy(output[idx_train], label[idx_train])
    loss_train.backward()
    optimizer.step()
    

    loss_val, acc_val = val(network, label, output, loss, idx_val)
    print('Epoch: {:03d} | Learning rate: {:.8f} | Train loss: {:.6f} | Train acc: {:.6f} | Val loss: {:.6f} | Val acc: {:.6f}'.\
    format(i+1, optimizer.param_groups[0]['lr'], loss_train.item(), acc_train.item(), loss_val.item(), acc_val.item()))
    losses.append(loss_train.item())
    train_accs.append(acc_train.item())
    val_accs.append(acc_val.item())

    early_stopping(loss_val, network)
    if early_stopping.early_stop:
        network.load_state_dict(torch.load("./chebynet.pth"))
        print('Early stopping.')
        break


# -----------------------
# 5. EVALUACIÓN FINAL
# -----------------------
with torch.no_grad():
    output_final = network(x_input, gso)  
    
    # Predecir clases
    pred_final = output_final.argmax(dim=1)  
    
    # Calcular accuracy en test
    test_acc = (pred_final[idx_test] == label[idx_test]).float().mean()
    print(f"\nACCURACY FINAL - Test: {test_acc:.4f}")

# También calcular accuracy en train y val para comparar
train_acc_final = (pred_final[idx_train] == label[idx_train]).float().mean()
val_acc_final = (pred_final[idx_val] == label[idx_val]).float().mean()

print(f"TRAIN ACCURACY - Final: {train_acc_final:.4f}")
print(f"VAL ACCURACY - Final: {val_acc_final:.4f}")

# Guardar resultados
np.savez_compressed(
    "resultados_clasificacion.npz",
    losses=losses,
    train_accs=train_accs, 
    val_accs=val_accs,
    test_acc=test_acc.item(),
    n_classes=n_class,
    n_features=n_feat
)