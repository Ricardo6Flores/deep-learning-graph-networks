# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 15:12:27 2025

@author: Ricar
"""


import networkx as nx
import random
import csv
import string
from Complejos_Simpliciales_mejorado import ComplejoSimplicial
import numpy as np
import os
# Generar grafo conexo
def generar_grafo_conexo(n, m, connectedness=0.5, max_degree=5, seed=None):
    """
    Genera un grafo conexo con n nodos y m aristas, opcionalmente limitando el grado máximo por nodo.
    
    Parámetros:
    - n: número de nodos
    - m: número total de aristas
    - connectedness: probabilidad de priorizar nodos con alto grado al agregar aristas
    - max_degree: límite máximo de aristas por nodo (None = sin límite)
    - seed: semilla para reproducibilidad
    """
    if m < n - 1:
        raise ValueError("El número de aristas debe ser al menos n-1 para garantizar conectividad.")
    
    rng = random.Random(seed)
    G = nx.random_tree(n, seed=seed)  # árbol inicial conexo
    
    # Lista de pares de nodos que aún no están conectados
    posibles_aristas = [(i, j) for i in range(n) for j in range(i+1, n) if not G.has_edge(i,j)]
    aristas_restantes = m - (n - 1)
    
    for _ in range(aristas_restantes):
        if not posibles_aristas:
            break
        
        # Filtrar pares que respeten max_degree
        if max_degree is not None:
            posibles_aristas_filtradas = [
                (u,v) for (u,v) in posibles_aristas 
                if G.degree(u) < max_degree and G.degree(v) < max_degree
            ]
            if not posibles_aristas_filtradas:
                break
        else:
            posibles_aristas_filtradas = posibles_aristas
        
        # Ordenar o mezclar según connectedness
        if rng.random() < connectedness:
            posibles_aristas_filtradas.sort(key=lambda x: G.degree(x[0]) + G.degree(x[1]), reverse=True)
        else:
            rng.shuffle(posibles_aristas_filtradas)
        
        # Agregar la primera arista válida
        arista = posibles_aristas_filtradas[0]
        G.add_edge(*arista)
        posibles_aristas.remove(arista)
    
    return G


import matplotlib.pyplot as plt
plt.rc('font', size=20)
# --------------------
# Función para graficar el grafo
def graficar_grafo_con_cocadenas_complejo(G, C, f0_array=None, f1_array=None, layout="spring", node_size=300, font_size=14, seed=None, guardar= None, mask_damaged_0=None):
    """
    Grafica el grafo mostrando cocadenas en los vértices y aristas, usando un ComplejoSimplicial.
    
    Parámetros:
    - G: grafo de networkx
    - C: ComplejoSimplicial
    - f0_array: array con cocadenas sobre los vértices, en el mismo orden que C.simplices_by_dim[0]
    - f1_array: array con cocadenas sobre las aristas, en el mismo orden que C.simplices_by_dim[1]
    """
    
    plt.figure(figsize=(12, 10))
    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    else:
        pos = nx.random_layout(G)
    
    # Dibujar aristas
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    

    # Dibujar nodos
    # --- Colorear nodos ---
    if mask_damaged_0 is not None:
        node_colors = ["red" if i in mask_damaged_0 else "skyblue" for i in G.nodes()]
    else:
        node_colors = "skyblue"

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, edgecolors="k")

    # Construir diccionario de cocadenas sobre vértices
    f0 = {}
    if f0_array is not None:
        for i, simplex in enumerate(C.simplices_by_dim.get(0, [])):
            v = list(simplex)[0]
            f0[v] = f0_array[i]
    
    # Construir diccionario de cocadenas sobre aristas
    f1 = {}
    if f1_array is not None:
        for i, simplex in enumerate(C.simplices_by_dim.get(1, [])):
            u, v = sorted(list(simplex))
            f1[(u,v)] = f1_array[i]
    
    # Etiquetas de nodos
    labels = {}
    for v in G.nodes():
        if f0_array is not None:
            labels[v] = f"{v}\n({f0[v]})"
        else:
            labels[v] = str(v)
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
    
    # Etiquetas de aristas
    if f1_array is not None:
        edge_labels = {}
        for u, v in G.edges():
            key = (u,v) if (u,v) in f1 else (v,u)
            if key in f1:
                edge_labels[(u,v)] = f"{f1[key]}"
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size-1)
    nx.draw_networkx_labels(G, pos, labels, font_size=font_size)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size-1)
    # --- Recuadro informativo fuera del gráfico ---
    if mask_damaged_0 is not None and len(mask_damaged_0) > 0:
        info = [f"Nodo {i} → Grado {G.degree(i)}" for i in mask_damaged_0]
        texto_info = f"{len(mask_damaged_0)} Nodos dañados:\n" + "\n".join(info)
        plt.gcf().text(
            1.02, 0.5, texto_info,
            fontsize=font_size - 1,
            color="red",
            va="center",
            ha="left",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", edgecolor="red", boxstyle="round,pad=0.5")
        )
    plt.axis("off")
    plt.title(f"Grafo con cocadenas\nVértices: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}", fontsize=25)
    plt.savefig(f"heatmaps/{guardar}/seed{seed}/{seed}.png", dpi=300, bbox_inches='tight')
    plt.show()


import pandas as pd

def mostrar_matriz_frontera(complejo, k):
    """
    Devuelve la matriz de frontera ∂ₖ como un DataFrame con etiquetas.
    Filas = (k-1)-símplices
    Columnas = k-símplices
    """
    B = complejo.construir_matriz_incidencia(k).toarray()
    
    # Etiquetas de filas y columnas
    filas = [tuple(sorted(s)) for s in complejo.simplices_by_dim.get(k-1, [])]
    columnas = [tuple(sorted(s)) for s in complejo.simplices_by_dim.get(k, [])]
    
    return pd.DataFrame(B, index=filas, columns=columnas)

def mostrar_matriz_laplaciano(complejo, k):
    """
    Devuelve la matriz de laplaciano como un DataFrame con etiquetas.
    Filas = (k-1)-símplices
    Columnas = k-símplices
    """
    L = complejo.construir_laplaciano(k).toarray()
    
    # Etiquetas de filas y columnas
    columnas = [tuple(sorted(s)) for s in complejo.simplices_by_dim.get(k, [])]
    
    return pd.DataFrame(L, index=columnas, columns=columnas)
   
    
# # --------------------
# # Ejemplo de uso
# #seed = 42
# seed = 42
# #G = generar_grafo_conexo(n=100, m=300, connectedness=0.7, seed=seed)
# #G = generar_grafo_conexo(n=10, m=20, connectedness=0.7, seed=seed)

# G = generar_grafo_conexo(n=20, m=40, connectedness=0.7, seed=seed)

# # Crear complejo simplicial vacío
# C = ComplejoSimplicial()

# # Añadir todos los vértices del grafo como 0-símplices
# for v in G.nodes():
#     C.agregar_vertice(v)

# # Añadir cada arista (u, v) como 1-símplice
# for u, v in G.edges():
#     C.agregar_simplex({u, v})
    
# print("Vértices:", C.k_simplices(0))
# print("Aristas:", C.k_simplices(1))
# print("Dimensión del complejo:", C.dimension())

# L0_df = mostrar_matriz_laplaciano(C, 0)
# print("Laplaciano 0, L0:\n", L0_df)
# L0 = C.construir_laplaciano(0).toarray()
# #print("Laplaciano L0:\n", L0)

# B1_df = mostrar_matriz_frontera(C, 1)
# print("Frontera 1, B1:\n", B1_df)


# B1 = C.construir_matriz_incidencia(1).toarray()


# # --------------------
# # Generar cocadenas enteras sobre aristas (1-símplices)
# aristas = C.k_simplices(1)
# np.random.seed(seed)
# f1_vector = np.random.randint(5, 16, size=len(aristas))  # enteros entre 5 y 15

# # Guardar cocadenas sobre aristas
# np.save("cocadenas_aristas.npy", f1_vector)
# print("Cocadenas sobre aristas guardadas en cocadenas_aristas.npy")
# print("Cocadenas aristas:", f1_vector)

# # --------------------
# # Propagar cocadenas a los vértices usando el coborde δ0*
# f0_vector = np.abs(B1).dot(f1_vector)

# # Guardar cocadenas sobre vértices
# np.save("cocadenas_vertices.npy", f0_vector)
# print("Cocadenas sobre vértices guardadas en cocadenas_vertices.npy")
# print("Cocadenas vértices:", f0_vector)

# # --------------------
# # Guardar matrices para referencia
# np.save("laplaciano_0.npy", L0)
# np.save("frontera_1.npy", B1)
# print("Matrices Laplaciano y Frontera guardadas en npy")


# # --------------------
# # Número de vértices a dañar
# num_damaged_vertices = 6
# np.random.seed(seed+1)

# # Elegir índices de vértices dañados
# mask_damaged_0 = np.random.choice(len(f0_vector), size=num_damaged_vertices, replace=False)
# np.save("mask_damaged_0.npy", mask_damaged_0)
# print("Máscara de vértices dañados guardada en mask_damaged_0.npy:", mask_damaged_0)

# # Máscara de vértices conocidos = complemento de los dañados
# mask_known_0 = np.array([i for i in range(len(f0_vector)) if i not in mask_damaged_0])
# np.save("mask_known_0.npy", mask_known_0)
# print("Máscara de vértices conocidos guardada en mask_known_0.npy:", mask_known_0)

# # --------------------
# # Cocadenas dañadas sobre vértices
# f0_damaged = f0_vector.copy()
# f0_damaged[mask_damaged_0] = 5  # asignar un valor pequeño
# np.save("cocadenas_vertices_danadas.npy", f0_damaged)
# print("Cocadenas vértices dañadas guardadas en cocadenas_vertices_danadas.npy:", f0_damaged)

# assert(np.array_equal(f0_damaged[mask_known_0], f0_vector[mask_known_0])) 
# # --------------------
# # Supongamos que ya generaste f1_array y f0_array
# graficar_grafo_con_cocadenas_complejo(G, C, f0_array=f0_vector, f1_array=f1_vector)



def grafo_malla(n=20, m=40, connectedness=0.7, seed=0, num_danados=6, guardar=None, grado = 5):
    """
    Genera un grafo conexo y guarda los archivos base necesarios
    para el pipeline de entrenamiento y análisis.

    Archivos generados:
    - laplaciano_0.npy
    - cocadenas_vertices.npy
    - cocadenas_vertices_danadas.npy
    - mask_known_0.npy
    - mask_damaged_0.npy
    """

    # Fijar semilla
    np.random.seed(seed)
    random.seed(seed)

    # === 1. Generar grafo conexo ===
    G = generar_grafo_conexo(n=n, m=m, connectedness=connectedness, seed=seed, max_degree= grado)

    # Crear complejo simplicial vacío
    C = ComplejoSimplicial()

    # Añadir todos los vértices del grafo como 0-símplices
    for v in G.nodes():
        C.agregar_vertice(v)

    # Añadir cada arista (u, v) como 1-símplice
    for u, v in G.edges():
        C.agregar_simplex({u, v})
        
    #print("Vértices:", C.k_simplices(0))
    #print("Aristas:", C.k_simplices(1))
    #print("Dimensión del complejo:", C.dimension())

    #L0_df = mostrar_matriz_laplaciano(C, 0)
    #print("Laplaciano 0, L0:\n", L0_df)
    L0 = C.construir_laplaciano(0).toarray()
    #print("Laplaciano L0:\n", L0)

    #B1_df = mostrar_matriz_frontera(C, 1)
    #print("Frontera 1, B1:\n", B1_df)


    B1 = C.construir_matriz_incidencia(1).toarray()


    # --------------------
    # Generar cocadenas enteras sobre aristas (1-símplices)
    aristas = C.k_simplices(1)
    np.random.seed(seed)
    f1_vector = np.random.randint(5, 16, size=len(aristas))  # enteros entre 5 y 15

    # Guardar cocadenas sobre aristas
    np.save("cocadenas_aristas.npy", f1_vector)
    #print("Cocadenas sobre aristas guardadas en cocadenas_aristas.npy")
    #print("Cocadenas aristas:", f1_vector)

    # --------------------
    # Propagar cocadenas a los vértices usando el coborde δ0*
    f0_vector = np.abs(B1).dot(f1_vector)

    # Guardar cocadenas sobre vértices
    np.save("cocadenas_vertices.npy", f0_vector)
    #print("Cocadenas sobre vértices guardadas en cocadenas_vertices.npy")
    #print("Cocadenas vértices:", f0_vector)

    # --------------------
    # Guardar matrices para referencia
    np.save("laplaciano_0.npy", L0)
    #np.save("frontera_1.npy", B1)
    #print("Matrices Laplaciano y Frontera guardadas en npy")


    # --------------------
    # Número de vértices a dañar
    num_damaged_vertices = num_danados
    # Obtener los grados
    grados = np.array([G.degree(i) for i in G.nodes()])

    # Definir probabilidades inversas al grado
    prob = 1 / (grados + 1e-6)  # evitar división por 0
    prob = prob / prob.sum()    # normalizar a suma 1

    # Elegir nodos dañados con prioridad a grados bajos
    # Seleccionar nodos sin que sean adyacentes
    max_intentos = 5000
    for intento in range(max_intentos):
        candidatos = np.random.choice(len(G.nodes()), size=num_damaged_vertices, replace=False, p=prob)
        # Verificar si hay aristas entre los elegidos
        es_valido = True
        for i in range(num_damaged_vertices):
            for j in range(i+1, num_damaged_vertices):
                if G.has_edge(candidatos[i], candidatos[j]):
                    es_valido = False
                    break
            if not es_valido:
                break
        if es_valido:
            mask_damaged_0 = candidatos
            break
    else:
        mask_damaged_0 = np.random.choice(len(G.nodes()), size=num_damaged_vertices, replace=False, p=prob)
        #raise ValueError("No se pudo encontrar un conjunto de nodos no adyacentes tras varios intentos")

    np.save("mask_damaged_0.npy", mask_damaged_0)
    print("Máscara de vértices dañados (baja+intermedia) guardada:", mask_damaged_0)

    # Máscara de nodos conocidos
    mask_known_0 = np.array([i for i in range(len(f0_vector)) if i not in mask_damaged_0])
    np.save("mask_known_0.npy", mask_known_0)
    print("Máscara de vértices conocidos guardada:", mask_known_0)

    # --------------------
    # Cocadenas dañadas sobre vértices
    f0_damaged = f0_vector.copy()
    minimo = np.min(f0_vector[mask_known_0])
    f0_damaged[mask_damaged_0] = minimo
    #f0_damaged[mask_damaged_0] = 5  # asignar un valor pequeño
    np.save("cocadenas_vertices_danadas.npy", f0_damaged)
    print("Cocadenas vértices dañadas guardadas en cocadenas_vertices_danadas.npy:", f0_damaged)

    assert(np.array_equal(f0_damaged[mask_known_0], f0_vector[mask_known_0])) 
    # --------------------
    graficar_grafo_con_cocadenas_complejo(G, C, f0_array=f0_vector, f1_array=f1_vector, seed=seed, guardar=guardar, mask_damaged_0=mask_damaged_0)
    
    
    
    # num_damaged_vertices = num_danados
    # np.random.seed(seed+1)

    # # Elegir índices de vértices dañados
    # mask_damaged_0 = np.random.choice(len(f0_vector), size=num_damaged_vertices, replace=False)
    # np.save("mask_damaged_0.npy", mask_damaged_0)
    # #print("Máscara de vértices dañados guardada en mask_damaged_0.npy:", mask_damaged_0)

    # # Máscara de vértices conocidos = complemento de los dañados
    # mask_known_0 = np.array([i for i in range(len(f0_vector)) if i not in mask_damaged_0])
    # np.save("mask_known_0.npy", mask_known_0)
    # #print("Máscara de vértices conocidos guardada en mask_known_0.npy:", mask_known_0)

    # # --------------------
    # # Cocadenas dañadas sobre vértices
    # f0_damaged = f0_vector.copy()
    # minimo = np.min(f0_vector[mask_known_0])
    # f0_damaged[mask_damaged_0] = minimo
    # np.save("cocadenas_vertices_danadas.npy", f0_damaged)
    # #print("Cocadenas vértices dañadas guardadas en cocadenas_vertices_danadas.npy:", f0_damaged)

    # assert(np.array_equal(f0_damaged[mask_known_0], f0_vector[mask_known_0])) 
    # # --------------------
    # # Supongamos que ya generaste f1_array y f0_array
    # graficar_grafo_con_cocadenas_complejo(G, C, f0_array=f0_vector, f1_array=f1_vector, seed=seed)

    return G



