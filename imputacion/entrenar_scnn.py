# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:40:23 2025

@author: Ricar
"""

#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from Complejos_Simpliciales_mejorado import SimplicialCNN, scipy_to_torch_sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse import coo_matrix


# -------------------------------------------------------------------------
# Escalar el laplaciano
# -------------------------------------------------------------------------
def escalar_laplaciano_por_espectro(L):
    """Escala el laplaciano dividiéndolo por su mayor autovalor."""
    lambda_max = eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    return L / (lambda_max + 1e-8)


# -------------------------------------------------------------------------
# Inicialización global (solo una vez)
# -------------------------------------------------------------------------
def preparar_datos():
    """Carga y prepara todos los datos fijos para los experimentos."""
    laplacian = np.load("laplaciano_0.npy", allow_pickle=True)
    f0 = np.load("cocadenas_vertices.npy", allow_pickle=True)
    f0_damaged = np.load("cocadenas_vertices_danadas.npy", allow_pickle=True)
    mask_known_0 = np.load("mask_known_0.npy", allow_pickle=True)
    mask_damaged_0 = np.load("mask_damaged_0.npy", allow_pickle=True)

    # Escalar y convertir el laplaciano
    Ls = scipy_to_torch_sparse(escalar_laplaciano_por_espectro(coo_matrix(laplacian)))

    eigvals = np.linalg.eigvalsh(coo_matrix(laplacian).toarray())
    mult_zero = np.sum(np.isclose(eigvals, 0, atol=1e-2))
    print(f"Multiplicidad de 0: {mult_zero}")

    batch_size = 1
    cochain_target = torch.tensor(f0, dtype=torch.float).view(batch_size, 1, -1)
    cochain_input = torch.tensor(f0_damaged, dtype=torch.float).view(batch_size, 1, -1)

    return {
        "Ls": Ls,
        "f0": f0,
        "mask_known_0": mask_known_0,
        "mask_damaged_0": mask_damaged_0,
        "cochain_target": cochain_target,
        "cochain_input": cochain_input,
    }


# -------------------------------------------------------------------------
# Entrenamiento de la red simplicial (usa datos precargados)
# -------------------------------------------------------------------------
def run_experiment(
    data_dict,
    N=5,
    hidden_features=2,
    total_iters=2000,
    learning_rate=0.001,
    save_prefix="resultados"
):
    """
    Entrena una red simplicial con parámetros dados, usando datos precargados.
    Guarda resultados y retorna la mejor pérdida.
    """

    torch.manual_seed(1337)
    np.random.seed(1337)

    Ls = data_dict["Ls"]
    f0 = data_dict["f0"]
    mask_known_0 = data_dict["mask_known_0"]
    mask_damaged_0 = data_dict["mask_damaged_0"]
    cochain_target = data_dict["cochain_target"]
    cochain_input = data_dict["cochain_input"]

    # Crear red y optimizador
    in_features, out_features = 1, 1
    network = SimplicialCNN(
        in_features=in_features,
        hidden_features=hidden_features,
        out_features=out_features,
        N=N,
        diagnostic=False,
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.L1Loss(reduction="sum")
    num_params = sum(np.prod(p.shape) for p in network.parameters())
    print(f"\n[N={N}, hidden={hidden_features}] -> {num_params} parámetros")

    all_outputs, all_predictions, all_actuals, losses = [], [], [], []

    for i in range(total_iters):
        optimizer.zero_grad()
        ys = network(cochain_input, Ls, mask_damaged_0)

        loss = criterion(ys[:, 0, mask_known_0], cochain_target[:, 0, mask_known_0])
        loss.backward()
        optimizer.step()

        out_full = ys.detach().cpu().numpy()[0, 0, :]
        pred_damaged = out_full[mask_damaged_0]
        true_damaged = f0[mask_damaged_0]

        all_outputs.append(out_full)
        all_predictions.append(pred_damaged)
        all_actuals.append(true_damaged)
        losses.append(loss.item())

        if i % 200 == 0 or i == total_iters - 1:
            print(f"[Iter {i+1}/{total_iters}] Loss = {loss.item():.6f}")

    # Guardar resultados
    save_name = f"Resultados/{save_prefix}_N{N}_hidden{hidden_features}.npz"
    np.savez_compressed(
        save_name,
        outputs=all_outputs,
        predictions=all_predictions,
        actuals=all_actuals,
        losses=losses,
        cochain_target=cochain_target.cpu().numpy(),
        N=N,
        total_iters=total_iters,
        learning_rate=learning_rate,
        hidden_features=hidden_features,
        num_params=num_params,
    )
    print(f"Resultados guardados en '{save_name}'")

    best_idx = np.argmin(losses)
    best_loss = losses[best_idx]
    return best_loss
