# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 18:37:37 2025

@author: Ricar
"""

import networkx as nx
import matplotlib.pyplot as plt
import xgi
import numpy as np
from matplotlib.cm import ScalarMappable 
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.colors as mcolors
from scipy.sparse import dok_matrix, lil_matrix
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from itertools import combinations
from typing import Set, List, Dict, FrozenSet
import scipy.sparse as sp


class ComplejoSimplicial:
    def __init__(self):
        self.vertices: Set = set()  # Conjunto de vértices
        self.simplices: Set[FrozenSet] = set()  # Conjunto de frozensets para símplices
        self._k_simplices_cache: Dict[int, List[FrozenSet]] = defaultdict(list)  # Cache para k-simplices
        self._dirty_flag = True  # Flag para indicar si la cache necesita actualización
        self.simplices_by_dim: Dict[int, List[FrozenSet]] = defaultdict(list)

    def agregar_vertice(self, vertice):
        """Añade un vértice al complejo de manera optimizada."""
        if vertice not in self.vertices:
            self.vertices.add(vertice)
            simplex = frozenset({vertice})
            if simplex not in self.simplices:
                self.simplices.add(simplex)
                self._k_simplices_cache[0].append(simplex)
                # Actualizamos simplices_by_dim[0] para mantener el orden
                if not hasattr(self, 'simplices_by_dim'):
                    self.simplices_by_dim = dict()
                if 0 not in self.simplices_by_dim:
                    self.simplices_by_dim[0] = []
                self.simplices_by_dim[0].append(simplex)

    
    def agregar_vertices(self, conjunto_vertices):
        """Añade un conjunto de vértices optimizado."""
        nuevos = set(conjunto_vertices) - self.vertices
        for v in nuevos:
            self.agregar_vertice(v)

    def agregar_simplex(self, simplex):
        """Añade un símplice y todas sus caras de manera optimizada, respetando el orden interno."""
        simplex = frozenset(simplex)
        if simplex in self.simplices:
            return
        
        # Agregar vértices nuevos
        nuevos_vertices = simplex - self.vertices
        for v in sorted(nuevos_vertices):  # opcional: orden fijo de vértices
            self.agregar_vertice(v)
        
        # Lista de símplices a agregar (incluye todas las caras)
        simplices_to_add = [simplex]
        for k in range(len(simplex)-1, 0, -1):
            # generar todas las caras de tamaño k
            for face in combinations(simplex, k):
                simplices_to_add.append(frozenset(face))
        
        # Agregar los símplices en orden
        for s in simplices_to_add:
            if s not in self.simplices:
                self.simplices.add(s)
                dim = len(s) - 1
                if dim not in self.simplices_by_dim:
                    self.simplices_by_dim[dim] = []
                self.simplices_by_dim[dim].append(s)

    def agregar_simplex_maximal(self, simplex):
        """
        Añade un símplice maximal al complejo:
        - Solo se agregan los vértices nuevos y el símplice dado.
        - No se recorren ni agregan automáticamente sus caras de menor dimensión.
        - Evita duplicados.
        - Actualiza _k_simplices_cache y simplices_by_dim.
        """
        simplex = frozenset(simplex)
        
        if simplex in self.simplices:
            return
        
        # Agregar al conjunto general
        self.simplices.add(simplex)
        
        # Dimensión del símplice
        k = len(simplex) - 1
        
        # Actualizar cache
        if k not in self._k_simplices_cache:
            self._k_simplices_cache[k] = []
        self._k_simplices_cache[k].append(simplex)
        
        # Actualizar simplices_by_dim
        if not hasattr(self, 'simplices_by_dim'):
            self.simplices_by_dim = defaultdict(list)
        self.simplices_by_dim[k].append(simplex)

    
    
    def agregar_simplices(self, *simplices):
        """Añade múltiples símplices optimizado."""
        for s in simplices:
            self.agregar_simplex(s)

    def obtener_vertices(self):
        # Asumimos que simplices_by_dim[0] guarda los vértices en orden
        return [next(iter(s)) for s in self.simplices_by_dim[0]]

    
    def obtener_simplices(self):
        # Devuelve todos los símplices en orden creciente de dimensión
        simplices_ordered = []
        for k in sorted(self.simplices_by_dim.keys()):
            simplices_ordered.extend(self.simplices_by_dim[k])
        return simplices_ordered

        
    def obtener_rango(self, simplex):
        return len(simplex) - 1
        
    def __str__(self):
        return f"Vértices: {self.vertices}\nSímplices: {len(self.simplices)} elementos"
        
    def grado_vertice(self, vertice):
        """Optimizado usando conteo directo"""
        return sum(vertice in s for s in self.simplices)
        
    def dimension(self):
        if not self.simplices:
            return -1
        return max(len(s) - 1 for s in self.simplices)
        
    def k_simplices(self, k: int) -> List[FrozenSet]:
        """Versión optimizada con caché, respetando el orden de inserción."""
        if not self._dirty_flag and k in self._k_simplices_cache:
            return self._k_simplices_cache[k]
        
        # Tomamos los símplices de la dimensión k según el orden de simplices_by_dim
        result = list(self.simplices_by_dim.get(k, []))
        self._k_simplices_cache[k] = result
        return result

        
    def etiquetar_simplices(self, k):
        """Devuelve un diccionario que mapea cada k-símplice a un índice único.
        
        Optimizaciones:
        - Usa frozenset directamente como clave cuando el orden no importa
        - Ordenamiento más eficiente para cuando se necesita orden
        - Evita recrear funciones lambda en cada llamada
        """
        def sort_key(x):
            return (isinstance(x, str), x)
        
        simplices = self.k_simplices(k)
        return {tuple(sorted(simplex, key=sort_key)): idx 
                for idx, simplex in enumerate(simplices)}



    
    def construir_matriz_incidencia(self, k):
        """Construye la matriz de borde ∂ₖ respetando el orden de simplices_by_dim"""
        if k == 0:
            raise ValueError("El operador de borde ∂₀ no está definido")
        
        simplices_k = self.simplices_by_dim.get(k, [])
        simplices_km1 = self.simplices_by_dim.get(k - 1, [])
        
        # Mapeo de índices según el orden de simplices_by_dim
        idx_km1 = {s: i for i, s in enumerate(simplices_km1)}
        
        B = lil_matrix((len(simplices_km1), len(simplices_k)), dtype=np.float32)
        
        for j, sigma in enumerate(simplices_k):
            sorted_sigma = sorted(sigma, key=lambda x: (isinstance(x, str), x))
            for i, cara in enumerate(combinations(sorted_sigma, k)):
                cara_fs = frozenset(cara)
                if cara_fs in idx_km1:
                    B[idx_km1[cara_fs], j] = (-1)**i
                    
        return B.tocsr()
    
    
    def construir_laplaciano(self, k):
        """Construye Lₖ = ∂ₖ₊₁ ∂ₖ₊₁* + ∂ₖ* ∂ₖ respetando el orden de simplices_by_dim"""
        if k == 0:
            B1 = self.construir_matriz_incidencia(1)
            L0 = B1.dot(B1.T)
            return L0.tocsr()
        else:
            Bk = self.construir_matriz_incidencia(k)
            try:
                Bkp1 = self.construir_matriz_incidencia(k + 1)
            except ValueError:
                # Si no hay símplices de dimensión k+1
                Bkp1 = dok_matrix((0, Bk.shape[1]), dtype=np.float32)
            
            Lk = Bk.T.dot(Bk) + Bkp1.dot(Bkp1.T)
            return Lk.tocsr()


    
    @staticmethod
    def scipy_to_torch_sparse(sparse_mtx):
        sparse_mtx = sparse_mtx.tocoo()
        indices = torch.from_numpy(
            np.vstack((sparse_mtx.row, sparse_mtx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mtx.data.astype(np.float32))
        shape = torch.Size(sparse_mtx.shape)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


    
    def es_subcomplejo(self, otro_complejo):
        """Verifica si este complejo es subcomplejo de otro."""
        if not self.vertices.issubset(otro_complejo.obtener_vertices()):
            return False
        return all(any(simplex == otro_s for otro_s in otro_complejo.obtener_simplices()) 
               for simplex in self.simplices)
               
    def crear_subcomplejo(self, vertices):
        """Crea un subcomplejo inducido por un conjunto de vértices."""
        if not vertices.issubset(self.vertices):
            raise ValueError("Algunos vértices no están en el complejo original")
            
        sub = ComplejoSimplicial()
        sub.agregar_vertices(vertices)
        
        for simplex in self.simplices:
            if simplex.issubset(vertices):
                sub.agregar_simplex(simplex)
                
        return sub
        
        
    def grafica(self, labels=False):
        """Visualiza el complejo simplicial."""
        def mapear_a_colores(lista_valores, colormap='viridis'):
            norm = mcolors.Normalize(vmin=min(lista_valores), vmax=max(lista_valores))
            cmap = plt.get_cmap(colormap)
            return [cmap(norm(valor)) for valor in lista_valores], norm, cmap
            
        lista_simplices = [list(simplex) for simplex in self.simplices]
        lista_valores = [self.obtener_rango(simplex) for simplex in self.simplices]
        
        color_, norm, cmap = mapear_a_colores(lista_valores)
        lista_hiperaristas_color = []
        lista_aristas_color = []
        lista_vertices_color = []
        
        for ind, color in enumerate(color_):
            if len(lista_simplices[ind]) == 1:
                lista_vertices_color.append(color)
            elif len(lista_simplices[ind]) == 2:
                lista_aristas_color.append(color)
            else:
                lista_hiperaristas_color.append(color)
                
        fig, ax = plt.subplots(figsize=(6, 2.5))
        H = xgi.Hypergraph()
        H.add_edges_from(lista_simplices)
        pos = xgi.barycenter_spring_layout(H, seed=1)
        
        if labels:
            ax, collections = xgi.draw(
                H,
                pos=pos,
                node_labels=True,
                node_size=10,
                hyperedge_labels=True,
                edge_fc=lista_hiperaristas_color,
                dyad_color=lista_aristas_color,
                node_fc=lista_vertices_color,
                hull=True,
            )
        else:
            ax, collections = xgi.draw(
                H,
                pos=pos,
                edge_fc=lista_hiperaristas_color,
                dyad_color=lista_aristas_color,
                node_fc=lista_vertices_color,
                hull=True,
            )
            
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Dimensión del símplice')
        
        plt.show()
       
########################################################
############## Clases del Modelo #######################
########################################################

def scipy_to_torch_sparse(sparse_mtx):
    sparse_mtx = sparse_mtx.tocoo()  # asegurar formato COO
    
    # indices más eficiente (2, nnz)
    indices = torch.from_numpy(
        np.vstack((sparse_mtx.row, sparse_mtx.col)).astype(np.int64)
    )
    
    # valores como float32
    values = torch.from_numpy(sparse_mtx.data.astype(np.float32))
    
    shape = torch.Size(sparse_mtx.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape, requires_grad=False)

def assemble(K, L, x):
    """
    Versión SIN batch dimension
    x shape: [C_in, M] donde:
      - C_in: número de características
      - M: número de nodos
    L shape: [M, M]
    """
    (C_in, M) = x.shape
    assert L.shape[0] == M
    assert L.shape[0] == L.shape[1]
    assert K > 0

    X = []
    for c_in in range(C_in):
        X_k = []
        # k=0
        if x.is_sparse:
            #print("x es disperso")
            x = x.to_dense()
        X_k.append(x[c_in, :].unsqueeze(1))  # [M, 1]
        
        # k=1
        if K > 1:
            Lx0 = L.mm(X_k[0])  # [M, 1]
            X_k.append(Lx0)
        
        # k >= 2
        for k in range(2, K):
            Xk = 2 * (L.mm(X_k[k-1])) - X_k[k-2]
            X_k.append(Xk)
        
        # Concatenar términos k: [M, K]
        X_k = torch.cat(X_k, 1)  
        X.append(X_k.unsqueeze(0))  # [1, M, K]

    X = torch.cat(X, 0)  # [C_in, M, K]
    assert X.shape == (C_in, M, K)
    return X


class SimplicialConvolutionPolynomialLayer(nn.Module):
    def __init__(self, in_features, out_features, variance=0.5, N=5, activation=nn.LeakyReLU()):
        super().__init__()
        self.N = N
        self.activation = activation
        self.weights = nn.parameter.Parameter(variance * torch.randn((out_features, in_features, self.N)))
        self.bias = nn.parameter.Parameter(torch.zeros((out_features, 1)))
        
        # Inicialización Xavier
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
  
    def forward(self, x, L):
        """
        x shape: [C_in, M] (características, nodos)
        L shape: [M, M]
        output shape: [C_out, M]
        """
        assert len(L.shape) == 2
        assert L.shape[0] == L.shape[1]
        
        (C_in, M) = x.shape
        assert M == L.shape[0]
    
        # X shape: [C_in, M, N]
        X = assemble(self.N, L, x)
        
        # Operación: y = sum_k X[:,:,:,k] * weights[:,:,k]
        # Resultado: [out_features, M]
        y = torch.einsum("imk,oik->om", (X, self.weights))
        
        return self.activation(y + self.bias) if self.activation else (y + self.bias)
    
class SimplicialCNN(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes, N):
        super().__init__()
        self.conv0_1 = SimplicialConvolutionPolynomialLayer(in_features, hidden_features, N=N)
        self.conv0_2 = SimplicialConvolutionPolynomialLayer(hidden_features, num_classes, N=N, activation=None)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X, L):
        """
        X shape: [M, C_in] = [2680, 302] (nodos, características)
        L shape: [M, M] = [2680, 2680]
        output shape: [M, num_classes] = [2680, 7]
        """
        # Transponer para: [C_in, M] = [302, 2680]
        X = X.T  # [302, 2680]
        
        X0_1 = self.conv0_1(X, L)  # [hidden_features, 2680]
        X0 = self.conv0_2(X0_1, L) # [num_classes, 2680]
        
        # Transponer de vuelta: [2680, num_classes]
        X0 = X0.T
        
        return self.log_softmax(X0)
        
