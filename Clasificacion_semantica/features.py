# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 00:10:37 2025

@author: Ricar
"""

import numpy as np
import pandas as pd
from numpy.linalg import norm
from difflib import SequenceMatcher

def generar_features_aristas(K, coords_en, coords_es, palabras_en, palabras_es,
                             cognados, falsos_cognados):
    """
    Genera features para CADA arista del complejo simplicial K
    y devuelve un DataFrame listo para guardar en CSV.
    """

    filas = []

    # --- Unir los embeddings en un solo diccionario ---
    todas_coords = {}
    for p, v in coords_en.items():
        todas_coords[p] = np.array(v)
    for p, v in coords_es.items():
        todas_coords[p] = np.array(v)

    # --- Crear lookup de cognados y falsos cognados ---
    set_cognados = {tuple(sorted(par)) for par in cognados}
    set_falsos = {tuple(sorted(par)) for par in falsos_cognados}

    # --- Obtener todas las aristas (1-simplices) ---
    aristas = K.k_simplices(1)   # lista de frozensets {p1,p2}

    for arista in aristas:
        p1, p2 = list(arista)

        v1 = todas_coords[p1]
        v2 = todas_coords[p2]

        # idiomas
        idioma1 = "en" if p1 in palabras_en else "es"
        idioma2 = "en" if p2 in palabras_en else "es"

        # distancia euclidiana
        dist = norm(v1 - v2)

        # coseno
        cos_sim = np.dot(v1, v2) / (norm(v1) * norm(v2))
        cos_dist = cos_sim

        # diferencia de normas
        norm_diff = abs(norm(v1) - norm(v2))

        # prefix boolean feat
        comparten_inicial = p1[0].lower() == p2[0].lower()

        # longitud
        diff_len = abs(len(p1) - len(p2))
        # --- Similitud ortográfica ---
        sim_ortografica = SequenceMatcher(None, p1, p2).ratio()
    
        # etiqueta por clases lingüísticas
        par = tuple(sorted((p1, p2)))
        if par in set_cognados:
            etiqueta = "C"
        elif par in set_falsos:
            etiqueta = "FC"
        else:
            etiqueta = "O"

        filas.append({
            "palabra1": p1,
            "palabra2": p2,
            "idioma1": idioma1,
            "idioma2": idioma2,
            "dist_euclidiana": dist,
            "dist_coseno": cos_dist,
            "sim_ort": sim_ortografica,
            "comparten_inicial": int(comparten_inicial),
            "diff_len": diff_len,
            "label": etiqueta
        })

    df = pd.DataFrame(filas)
    df.to_csv("features_aristas.csv", index=False)
    print("CSV guardado en features_aristas.csv")

    return df
