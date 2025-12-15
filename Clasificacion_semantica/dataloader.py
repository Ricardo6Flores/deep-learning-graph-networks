import os
import numpy as np
import scipy.sparse as sp

def norm_feat(feature):
    feature = feature.astype(dtype=np.float32)
    if sp.issparse(feature):
        row_sum = feature.sum(axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = sp.diags(row_sum_inv, format='csc')
        norm_feature = deg_inv.dot(feature)
    else:
        row_sum_inv = np.power(np.sum(feature, axis=1), -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        norm_feature = deg_inv.dot(feature)
        norm_feature = np.array(norm_feature, dtype=np.float32)

    return norm_feature

def load_data(train_size=0.5, valid_size=0.2, test_size=0.3, random_seed=42):
    """
    Carga features y labels desde features_aristas.csv generados previamente.
    Usa columnas 5, 6, 7 y 9 como features.
    Última columna = label (string) que se mapea a {0,1,2}.
    
    También carga L1.npz y genera idx_train, idx_valid, idx_test.
    """

    np.random.seed(random_seed)

    # ================================
    # 1) Cargar CSV de características
    # ================================
    csv_path = "features_aristas.csv"
    data = np.genfromtxt(csv_path, delimiter=",", dtype=str)

    # Eliminamos fila de encabezados si existe (checar si contiene letras)
    if not np.char.isnumeric(data[0]).all():
        data = data[1:]

    # Convertir los valores numéricos (features)
    # columnas 5,6,7,9 → índices 4,5,6,8 (0-based)
    feature_columns = [6]
    feature = data[:, feature_columns].astype(np.float32)
    n_feat = feature.shape[1]

    # ====================
    # 2) Procesar etiquetas
    # ====================
    etiquetas_raw = data[:, -1]   # última columna como string

    label = np.zeros(len(etiquetas_raw), dtype=np.int64)

    for i, v in enumerate(etiquetas_raw):
        v = v.strip().lower()
        if v == "c":
            label[i] = 0
        elif v == "fc":
            label[i] = 1
        else:
            label[i] = 2   # nada

    n_class = 3

    # ================================
    # 3) Cargar Laplaciano L1
    # ================================
    L1_path = "L1.npz"
    L1 = sp.load_npz(L1_path)

    # 4) Generar partición train/valid/test 
    N = len(label) 
    idx = np.arange(N) 
    np.random.shuffle(idx) 
    test_split = int(N * test_size) 
    valid_split = int(N * valid_size) 
    idx_test = idx[:test_split] 
    idx_valid = idx[test_split:test_split + valid_split] 
    idx_train = idx[test_split + valid_split:] 
    #proporciones_por_conjunto(label, idx_train, idx_valid, idx_test)

    # ========================== # 5) Retornar todo # ========================== 
    return (feature, L1, label, idx_train, idx_valid, idx_test, n_feat, n_class)


def proporciones_por_conjunto(label, idx_train, idx_valid, idx_test):
    import numpy as np
    
    def contar(y):
        # Cuenta cuántos hay de cada clase
        valores, counts = np.unique(y, return_counts=True)
        total = len(y)
        proporciones = {int(v): counts[i] / total for i, v in enumerate(valores)}
        return counts, proporciones

    # Cálculo por conjunto
    counts_total, prop_total = contar(label)
    counts_train, prop_train = contar(label[idx_train])
    counts_valid, prop_valid = contar(label[idx_valid])
    counts_test, prop_test = contar(label[idx_test])

    print("\n====== DISTRIBUCIÓN DE CLASES ======")

    print("\nTOTAL:")
    print("  Cantidades:", counts_total)
    print("  Proporciones:", prop_total)

    print("\nTRAIN:")
    print("  Cantidades:", counts_train)
    print("  Proporciones:", prop_train)

    print("\nVALID:")
    print("  Cantidades:", counts_valid)
    print("  Proporciones:", prop_valid)

    print("\nTEST:")
    print("  Cantidades:", counts_test)
    print("  Proporciones:", prop_test)
    
    return {
        "total": prop_total,
        "train": prop_train,
        "valid": prop_valid,
        "test": prop_test
    }
resp = load_data()