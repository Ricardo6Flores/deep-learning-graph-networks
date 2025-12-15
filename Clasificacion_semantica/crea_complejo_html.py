# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 16:13:34 2025

@author: Ricar
"""

import pickle
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import webbrowser
import os
import itertools
from scipy.spatial.distance import pdist, squareform, cosine
from adyacencia_grafo import crear_matriz_adyacencia_y_grafo
from adyacencia_grafo import analizar_estadisticas_grafo
import Complejos_Simpliciales


def cargar_embeddings():
    print("Cargando embeddings...")
    
    try:
        with open('embeddings_completos.pkl', 'rb') as f:
            datos = pickle.load(f)
        
        embeddings_en = datos['en']
        embeddings_es = datos['es']
        palabras_en = datos['palabras_en']
        palabras_es = datos['palabras_es']
        cognados = datos['cognados']
        falsos_cognados = datos['falsos_cognados']
        
        print(f"  Datos cargados:")
        print(f"  • Palabras inglés: {len(palabras_en)}")
        print(f"  • Palabras español: {len(palabras_es)}")
        print(f"  • Cognados: {len(cognados)}")
        print(f"  • Falsos cognados: {len(falsos_cognados)}")
        
        return embeddings_en, embeddings_es, palabras_en, palabras_es, cognados, falsos_cognados
    
    except FileNotFoundError:
        print("   Archivo 'embeddings_completos.pkl' no encontrado")
        print("   Ejecutar primero el script crea_palabras.py")
        return None


def reducir_dimensionalidad(embeddings_en, embeddings_es, metodo='pca'):
    """
    Reduce embeddings de 384D a 3D para visualización
    """
    print(f"\n🧮 Reduciendo dimensionalidad usando {metodo.upper()}...")
    
    # Combinar todos los embeddings
    todas_palabras = list(embeddings_en.keys()) + list(embeddings_es.keys())
    todas_embeddings = []
    
    for palabra in todas_palabras:
        if palabra in embeddings_en:
            todas_embeddings.append(embeddings_en[palabra])
        else:
            todas_embeddings.append(embeddings_es[palabra])
    
    X = np.array(todas_embeddings)
    print(f"  Matriz de embeddings: {X.shape}")
    
    # Aplicar reducción de dimensionalidad
    if metodo.lower() == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        X_3d = reducer.fit_transform(X)
        print(f"  Varianza explicada: {sum(reducer.explained_variance_ratio_):.2%}")
        
    elif metodo.lower() == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=30, 
                      n_iter=1000, learning_rate=200)
        X_3d = reducer.fit_transform(X)
    
    else:
        raise ValueError(f"Método {metodo} no reconocido")
    
    # Separar de nuevo por idioma
    coords_en = {}
    coords_es = {}
    
    for i, palabra in enumerate(todas_palabras):
        if i < len(embeddings_en):
            coords_en[palabra] = X_3d[i]
        else:
            coords_es[palabra] = X_3d[i]
    
    return coords_en, coords_es, X_3d, todas_palabras


def calcular_conexiones_epsilon(coords_en, coords_es, epsilon):
    """
    Calcula conexiones entre vértices basadas en distancia coseno.
    Retorna listas de conexiones intra-idioma e inter-idioma.
    """
    print(f"\n Calculando conexiones con epsilon={epsilon}...")
    
    # Combinar todas las coordenadas
    todas_coords = {}
    todas_coords.update(coords_en)
    todas_coords.update(coords_es)
    
    # Crear lista de palabras y sus coordenadas
    palabras = list(todas_coords.keys())
    coords_array = np.array([todas_coords[p] for p in palabras])
    
    # Calcular matriz de distancias
    dist_matrix = squareform(pdist(coords_array, metric='cosine'))
    
    n = len(palabras)
    # Diccionario para almacenar las mejores conexiones por palabra
    mejores_conexiones = {palabra: [] for palabra in palabras}
    
    # Primero recopilar todas las posibles conexiones
    # Contadores para diagnóstico
    count_es = 0
    count_en = 0
    count_mixtas = 0      #cuántas intentó conectar entre idiomas
    
    for i in range(n):
        for j in range(i+1, n):
    
            palabra1 = palabras[i]
            palabra2 = palabras[j]
    
            distancia = dist_matrix[i, j]
    
            # Determinar idiomas
            idioma1 = 'en' if palabra1 in coords_en else 'es'
            idioma2 = 'en' if palabra2 in coords_en else 'es'
    
            # === APLICACIÓN DE EPSILONS DIFERENCIADOS ===
            if idioma1 == 'en' and idioma2 == 'en':
                umbral = epsilon      # por ejemplo epsilon_en = 0.20 + 0.2
            elif idioma1 == 'es' and idioma2 == 'es':
                umbral = epsilon      # por ejemplo epsilon_es = 0.20 
            else:
                count_mixtas += 1
                continue
    
            # Si la distancia está dentro del umbral asignado
            if distancia < umbral:
    
                # **sumar al contador correspondiente**
                if idioma1 == 'es': 
                    count_es += 1
                else:
                    count_en += 1
    
                # guardar la conexión
                mejores_conexiones[palabra1].append((palabra2, distancia, True))
                mejores_conexiones[palabra2].append((palabra1, distancia, True))
    
    # === imprimir diagnóstico ===
    print(f" Conexiones candidatas encontradas:")
    print(f"   • Español–Español: {count_es}")
    print(f"   • Inglés–Inglés:   {count_en}")
    print(f"   • Mixtas ignoradas: {count_mixtas}")


    
    # Ordenar por distancia y tomar solo las n mejores por palabra
    conexiones_procesadas = set()  # Para evitar duplicados
    conexiones_intra = []  # Conexiones dentro del mismo idioma
    conexiones_inter = []  # Conexiones entre idiomas diferentes
    
    for palabra in palabras:
    
        # Filtrar conexiones de esta palabra
        conexiones_palabra = mejores_conexiones[palabra]
    
        # Dividir conexiones por tipo
        conexiones_es = []
        conexiones_en = []
        
        for otra_palabra, distancia, mismo_idioma in conexiones_palabra:
            if palabra in coords_es:
                if otra_palabra in coords_es:
                    conexiones_es.append((otra_palabra, distancia))
            else:  # palabra está en inglés
                if otra_palabra in coords_en:
                    conexiones_en.append((otra_palabra, distancia))
    
        # Ordenar por distancia
        conexiones_es = sorted(conexiones_es, key=lambda x: x[1])
        conexiones_en = sorted(conexiones_en, key=lambda x: x[1])
    
        # Elegir las 4 mejores según idioma de la palabra
        if palabra in coords_es:
            mejores = conexiones_es[:3]
        else:
            mejores = conexiones_en[:3]
    
        # Agregar al grafo
        for otra_palabra, distancia in mejores:
            conexion = tuple(sorted([palabra, otra_palabra])) + (distancia,)
            
            if conexion not in conexiones_procesadas:
                conexiones_procesadas.add(conexion)
                
                if (palabra in coords_es and otra_palabra in coords_es) or \
                   (palabra in coords_en and otra_palabra in coords_en):
                    conexiones_intra.append((palabra, otra_palabra, distancia))
                else:
                    conexiones_inter.append((palabra, otra_palabra, distancia))

        
  
    print(f"    • Intra-idioma: {len(conexiones_intra)}")
    print(f"    • Inter-idioma: {len(conexiones_inter)}")
    
    # === Construcción de TRIÁNGULOS (2-simplejos) ===
    # Necesitamos las aristas intra-idioma como conjunto para acceso rápido
    aristas_intra_set = set()
    for a, b, d in conexiones_intra:
        aristas_intra_set.add(tuple(sorted([a, b])))
    
    triangulos = []
    
    # Recorremos todas las tripletas posibles dentro del mismo idioma
    # Para que sea eficiente, agrupamos palabras por idioma
    palabras_en = [p for p in palabras if p in coords_en]
    palabras_es = [p for p in palabras if p in coords_es]
    
    def construir_triangulos(lista_palabras):
        t = []
        m = len(lista_palabras)
        for i in range(m):
            for j in range(i+1, m):
                for k in range(j+1, m):
                    p1, p2, p3 = lista_palabras[i], lista_palabras[j], lista_palabras[k]
                    e12 = tuple(sorted([p1, p2]))
                    e13 = tuple(sorted([p1, p3]))
                    e23 = tuple(sorted([p2, p3]))
                    # Si las 3 aristas existen → triángulo
                    if e12 in aristas_intra_set and e13 in aristas_intra_set and e23 in aristas_intra_set:
                        t.append((p1, p2, p3))
        return t
    
    triangulos_en = construir_triangulos(palabras_en)
    triangulos_es = construir_triangulos(palabras_es)
    
    triangulos = triangulos_en + triangulos_es
    
    print(f"    • Triángulos EN: {len(triangulos_en)}")
    print(f"    • Triángulos ES: {len(triangulos_es)}")
    print(f"    • Total triángulos: {len(triangulos)}")

    return conexiones_intra, conexiones_inter, palabras, coords_array, triangulos

def crear_visualizacion_3d_epsilon(coords_en, coords_es, palabras_en, palabras_es, 
                                  cognados, falsos_cognados, metodo='pca', epsilon=1.0):
    """
    Crea visualización 3D interactiva con conexiones basadas en epsilon
    """
    
    # Calcular conexiones basadas en epsilon
    conexiones_intra, conexiones_inter, todas_palabras, todas_coords, triangulos = calcular_conexiones_epsilon(
        coords_en, coords_es, epsilon
    )
    
    K = Complejos_Simpliciales.ComplejoSimplicial()



    #Agregar triángulos (2-símplices)
    for p1, p2, p3 in triangulos:
        K.agregar_simplex({p1, p2, p3})
        
    #Agregar aristas intra-idioma
    for p1, p2, dist in conexiones_intra:
        K.agregar_simplex({p1, p2}) # arista
    
    for p_en, p_es in cognados:
        K.agregar_simplex({p_en, p_es})
    
    for p_en, p_es in falsos_cognados:
        K.agregar_simplex({p_en, p_es})
    
    L1 = K.construir_laplaciano(1)
    import scipy.sparse as sp

    sp.save_npz("L1.npz", L1)


    import features
    
    features.generar_features_aristas(K, coords_en, coords_es, palabras_en, palabras_es,
                                 cognados, falsos_cognados)
    ##############################################################################################
    adj_matrix, G = crear_matriz_adyacencia_y_grafo(todas_palabras, todas_coords, conexiones_intra, conexiones_inter, 
                                       coords_en, coords_es, epsilon)
    
    df_estadisticas, nodos_bajo_grado = analizar_estadisticas_grafo(G, palabras_en, palabras_es, epsilon)
    ##############################################################################################
    
    # Crear figura
    fig = go.Figure()
    
    # Diccionario para mapear palabras a índices
    palabra_a_indice = {palabra: i for i, palabra in enumerate(todas_palabras)}
    
    # 1. Puntos para palabras en inglés
    x_en, y_en, z_en = [], [], []
    textos_en = []
    indices_en = []
    
    # 3. Reducir dimensionalidad
    coords_en, coords_es, _, _ = reducir_dimensionalidad(
        coords_en, coords_es, metodo='pca'
    )
    
    for palabra in palabras_en:
        if palabra in coords_en:
            x, y, z = coords_en[palabra]
            x_en.append(x)
            y_en.append(y)
            z_en.append(z)
            textos_en.append(f"🇬🇧 {palabra}")
            indices_en.append(palabra_a_indice[palabra])
    
    fig.add_trace(go.Scatter3d(
        x=x_en, y=y_en, z=z_en,
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.8,
            symbol='circle',
            line=dict(width=1, color='darkblue')
        ),
        name=f'Inglés ({len(x_en)} palabras)',
        text=textos_en,
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # 2. Puntos para palabras en español
    x_es, y_es, z_es = [], [], []
    textos_es = []
    indices_es = []
    
    for palabra in palabras_es:
        if palabra in coords_es:
            x, y, z = coords_es[palabra]
            x_es.append(x)
            y_es.append(y)
            z_es.append(z)
            textos_es.append(f"🇪🇸 {palabra}")
            indices_es.append(palabra_a_indice[palabra])
    
    fig.add_trace(go.Scatter3d(
        x=x_es, y=y_es, z=z_es,
        mode='markers',
        marker=dict(
            size=8,
            color='red',
            opacity=0.8,
            symbol='diamond',
            line=dict(width=1, color='darkred')
        ),
        name=f'Español ({len(x_es)} palabras)',
        text=textos_es,
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))
    
    # 3. Líneas para conexiones intra-idioma (inglés-inglés y español-español)
    lineas_intra_x, lineas_intra_y, lineas_intra_z = [], [], []
    for palabra1, palabra2, distancia in conexiones_intra:
        if palabra1 in coords_en and palabra2 in coords_en:
            x1, y1, z1 = coords_en[palabra1]
            x2, y2, z2 = coords_en[palabra2]
            lineas_intra_x.extend([x1, x2, None])
            lineas_intra_y.extend([y1, y2, None])
            lineas_intra_z.extend([z1, z2, None])
        elif palabra1 in coords_es and palabra2 in coords_es:
            x1, y1, z1 = coords_es[palabra1]
            x2, y2, z2 = coords_es[palabra2]
            lineas_intra_x.extend([x1, x2, None])
            lineas_intra_y.extend([y1, y2, None])
            lineas_intra_z.extend([z1, z2, None])
    
    if lineas_intra_x:
        fig.add_trace(go.Scatter3d(
            x=lineas_intra_x, y=lineas_intra_y, z=lineas_intra_z,
            mode='lines',
            line=dict(color='purple', width=2),
            name=f'Conexiones intra-idioma (ε={epsilon})',
            hoverinfo='none',
            opacity=0.8
        ))
    
    # # 4. Líneas para conexiones inter-idioma
    # lineas_inter_x, lineas_inter_y, lineas_inter_z = [], [], []
    # for palabra1, palabra2, distancia in conexiones_inter:
    #     if (palabra1 in coords_en and palabra2 in coords_es) or (palabra1 in coords_es and palabra2 in coords_en):
    #         if palabra1 in coords_en:
    #             x1, y1, z1 = coords_en[palabra1]
    #             x2, y2, z2 = coords_es[palabra2]
    #         else:
    #             x1, y1, z1 = coords_es[palabra1]
    #             x2, y2, z2 = coords_en[palabra2]
            
    #         lineas_inter_x.extend([x1, x2, None])
    #         lineas_inter_y.extend([y1, y2, None])
    #         lineas_inter_z.extend([z1, z2, None])
    
    # if lineas_inter_x:
    #     fig.add_trace(go.Scatter3d(
    #         x=lineas_inter_x, y=lineas_inter_y, z=lineas_inter_z,
    #         mode='lines',
    #         line=dict(color='purple', width=2),
    #         name=f'Conexiones inter-idioma (ε={epsilon})',
    #         hoverinfo='none',
    #         opacity=0.7
    #     ))
    
    # 5. Líneas para cognados verdaderos
    lineas_cognados_x, lineas_cognados_y, lineas_cognados_z = [], [], []
    for en, es in cognados:
        if en in coords_en and es in coords_es:
            x1, y1, z1 = coords_en[en]
            x2, y2, z2 = coords_es[es]
            
            lineas_cognados_x.extend([x1, x2, None])
            lineas_cognados_y.extend([y1, y2, None])
            lineas_cognados_z.extend([z1, z2, None])
    
    if lineas_cognados_x:
        fig.add_trace(go.Scatter3d(
            x=lineas_cognados_x, y=lineas_cognados_y, z=lineas_cognados_z,
            mode='lines',
            line=dict(color='green', width=3),
            name='Cognados verdaderos',
            hoverinfo='none'
        ))
    
    # 6. Líneas para falsos cognados
    lineas_falsos_x, lineas_falsos_y, lineas_falsos_z = [], [], []
    for en, es in falsos_cognados:
        if en in coords_en and es in coords_es:
            x1, y1, z1 = coords_en[en]
            x2, y2, z2 = coords_es[es]
            
            lineas_falsos_x.extend([x1, x2, None])
            lineas_falsos_y.extend([y1, y2, None])
            lineas_falsos_z.extend([z1, z2, None])
    
    if lineas_falsos_x:
        fig.add_trace(go.Scatter3d(
            x=lineas_falsos_x, y=lineas_falsos_y, z=lineas_falsos_z,
            mode='lines',
            line=dict(color='orange', width=3),
            name='Falsos cognados',
            hoverinfo='none'
        ))
    
    # 7. Configurar layout
    fig.update_layout(
        title=dict(
            text=f'Visualización 3D - {metodo.upper()} (ε={epsilon})',
            font=dict(size=20),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(title='Dimensión 1', showticklabels=False),
            yaxis=dict(title='Dimensión 2', showticklabels=False),
            zaxis=dict(title='Dimensión 3', showticklabels=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=10)
        ),
        width=1400,
        height=800,
        hovermode='closest'
    )
    
    # 8. Agregar controles interactivos
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.1,
                y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Mostrar Todo",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, True]}]
                    ),
                    dict(
                        label="Solo Puntos",
                        method="update",
                        args=[{"visible": [True, True, False, False, False, False]}]
                    ),
                    dict(
                        label="Solo Conexiones ε",
                        method="update",
                        args=[{"visible": [False, False, True, True, False, False]}]
                    ),
                    dict(
                        label="Solo Cognados",
                        method="update",
                        args=[{"visible": [False, False, False, False, True, True]}]
                    ),
                    dict(
                        label="Comparación",
                        method="update",
                        args=[{"visible": [True, True, True, True, True, False]}]
                    ),
                ]
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.3,
                y=1.15,
                showactive=True,
                buttons=[
                    dict(
                        label="Vista 3D",
                        method="relayout",
                        args=["scene.camera", dict(eye=dict(x=1.5, y=1.5, z=1.5))]
                    ),
                    dict(
                        label="Vista XY",
                        method="relayout",
                        args=["scene.camera", dict(eye=dict(x=0, y=0, z=2.5))]
                    ),
                    dict(
                        label="Vista XZ",
                        method="relayout",
                        args=["scene.camera", dict(eye=dict(x=0, y=2.5, z=0))]
                    ),
                    dict(
                        label="Vista YZ",
                        method="relayout",
                        args=["scene.camera", dict(eye=dict(x=2.5, y=0, z=0))]
                    ),
                ]
            )
        ]
    )
    
    # 9. Agregar anotaciones estadísticas
    fig.add_annotation(
        x=0.02, y=0.02,
        xref="paper", yref="paper",
        text=f"ε={epsilon}<br>Intra: {len(conexiones_intra)}<br>Inter: {len(conexiones_inter)}",
        showarrow=False,
        font=dict(size=12, color="black"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig, conexiones_intra, conexiones_inter, adj_matrix, G, df_estadisticas, nodos_bajo_grado, L1

def generar_conexiones_epsilon(coords_en, coords_es, palabras_en, palabras_es, 
                           cognados, falsos_cognados, metodo='pca'):
    """
    Genera múltiples visualizaciones con diferentes valores de epsilon
    """
    print("\n" + "=" * 60)
    
    # Valor de epsilon a probar
    epsilon = 0.55
   
    print(f" Generando visualización con ε={epsilon}")
    
    # Crear figura con el epsilon actual
    fig, conexiones_intra, conexiones_inter, adj_matrix, G, df_estadisticas, nodos_bajo_grado, L1 = crear_visualizacion_3d_epsilon(
        coords_en, coords_es, palabras_en, palabras_es,
        cognados, falsos_cognados, metodo=metodo, epsilon=epsilon
    )
    
    # Guardar archivo
    nombre_archivo = f'embeddings_3d_{metodo}_epsilon_{epsilon:.2f}.html'
    archivo_guardado = guardar_como_html(fig, nombre_archivo=nombre_archivo)

    
    return archivo_guardado, adj_matrix, df_estadisticas, L1

def guardar_como_html(fig, nombre_archivo='embeddings_3d.html'):
    """
    Guarda la figura como HTML interactivo
    """
    print(f" Guardando como HTML: {nombre_archivo}")
    
    # Configuración para HTML
    config = {
        'displayModeBar': True,
        'scrollZoom': True,
        'displaylogo': False,
        'modeBarButtonsToAdd': [
            'drawline',
            'drawopenpath',
            'eraseshape'
        ]
    }
    
    # Guardar
    fig.write_html(
        nombre_archivo,
        config=config,
        include_plotlyjs='cdn',
        full_html=True,
        auto_open=False
    )
    
    return nombre_archivo


print("=" * 70)
print("VISUALIZACIÓN DE EMBEDDINGS")
print("=" * 70)

#Cargar datos
embeddings_en, embeddings_es, palabras_en, palabras_es, cognados, falsos_cognados = cargar_embeddings()

archivo_epsilon, matriz_adj, estadisticas, L1 = generar_conexiones_epsilon(
    embeddings_en, embeddings_es, palabras_en, palabras_es,
    cognados, falsos_cognados, metodo='pca'
)
L1_densa = L1.toarray()

print("\n" + "=" * 70)
print("🎯 VISUALIZACIONES LISTAS")
print("=" * 70)
print(f"Archivo creado ({archivo_epsilon}):")

# Buscar archivo de resumen
archivos_html = [f for f in os.listdir() if f.startswith('resumen_epsilon_') and f.endswith('.html')]
if archivos_html:
    print(f"\n📊 Resumen generado: {archivos_html[0]}")









