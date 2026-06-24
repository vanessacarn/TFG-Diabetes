# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:30:52 2026

@author: vanes
"""

import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (cohen_kappa_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, ConfusionMatrixDisplay, average_precision_score, fbeta_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.base import clone  
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight
 
# =============================================================================
# 1. CARGA Y FUSIÓN DE CICLOS
# =============================================================================
data_dir = "NHANES_data"

def load_cycle(letra):
    datasets = {}
    
    for file in glob.glob(f"{data_dir}/*_{letra}.XPT"):
        name = os.path.basename(file).split("_")[0]
        datasets[name] = pd.read_sas(file, format="xport", encoding="latin-1")
    
    return datasets


def merge_cycle(datasets):
    df = datasets.get("DEMO")
    if df is None:
        raise ValueError("DEMO dataset missing")
    
    for key in datasets:
        if key != "DEMO":
            cols_to_use = datasets[key].columns.difference(df.columns).tolist()

            if "SEQN" not in cols_to_use:
                cols_to_use.append("SEQN")
            
            # Unimos solo las columnas que no están repetidas
            df = df.merge(datasets[key][cols_to_use], on="SEQN", how="left")
            
    return df 


data_H = merge_cycle(load_cycle("H"))
data_I = merge_cycle(load_cycle("I"))
data_P = merge_cycle(load_cycle("P"))
data_G = merge_cycle(load_cycle("G"))
data_F = merge_cycle(load_cycle("F"))
data_E = merge_cycle(load_cycle("E"))

nhanes = pd.concat([data_H, data_I, data_P, data_G, data_F, data_E], ignore_index=True)


# =============================================================================
# 2. SELECCIÓN DE VARIABLES DE INTERÉS
# =============================================================================
# Lista de las variables de interes
variables = [
    "SEQN",
    "RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "DMDEDUC2",
    "BPXSY1", "BPXDI1", "BMXBMI", "BMXWAIST",
    "LBXTR", "LBXTC", "LBXIN", "LBXSATSI",
    "LBXGLU", "LBXGH", "LBDHDD", "LBDLDL",
    "URDACT", "LBXSCR", "LBXSUA", 
    "ALQ101", "PAQ650", "PAD680",
    "SLQ050", "SLQ120", "SMQ020",
    "DBD900", "DPQ010", "HSD010", "FSDHH", 
    "MCQ300C", "DIQ010"
]

# Crear el nuevo DataFrame con solo esas columnas
datos = nhanes[variables].copy()
print(f"Dimensiones iniciales: {datos.shape}")


# =============================================================================
# 3. RESTRICCIÓN A ADULTOS (>= 18 años)
# =============================================================================
datos = datos[datos["RIDAGEYR"] >= 18].copy()
print(f"Tras filtrar adultos (>=18): {datos.shape}")


# =============================================================================
# 4. ARREGLO DE VARIABLES
# =============================================================================
# RIDRETH3 tiene un 28% de NAs porque los ciclos más antiguos usan RIDRETH1. Asi que
# usamos RIDRETH3 donde existe y RIDRETH1 donde no.
datos = datos.merge(nhanes[["SEQN", "RIDRETH1"]], on="SEQN", how="left")
datos["RIDRETH1_mapped"] = datos["RIDRETH1"].replace({5: 7})
datos["ETNIA"] = datos["RIDRETH3"].fillna(datos["RIDRETH1_mapped"])
datos = datos.drop(columns=["RIDRETH1", "RIDRETH1_mapped", "RIDRETH3"])

# Reconstrucción de URDACT donde falta: URDACT = albúmina urinaria (µg/mL) / creatinina urinaria (mg/dL) 
datos = datos.merge(nhanes[["SEQN", "URXUMA", "URXUCR"]], on="SEQN", how="left")
datos["URDACT_calc"] = datos["URXUMA"] / datos["URXUCR"]
datos["URDACT_calc"] = datos["URDACT_calc"].where(datos["URDACT_calc"].between(0, 10000))
datos["URDACT"] = datos["URDACT"].fillna(datos["URDACT_calc"])
datos = datos.drop(columns=["URXUMA", "URXUCR", "URDACT_calc"])

# Fusión de presión arterial: ciclos E-I usan BPXDI1/BPXSY1, ciclo P usa BPXODI1/BPXOSY1
datos = datos.merge(nhanes[["SEQN", "BPXODI1", "BPXOSY1"]], on="SEQN", how="left")
datos["BPXDI1"] = datos["BPXDI1"].combine_first(datos["BPXODI1"])
datos["BPXSY1"] = datos["BPXSY1"].combine_first(datos["BPXOSY1"])
datos = datos.drop(columns=["BPXODI1", "BPXOSY1"])


# =============================================================================
# 5. CÓDIGOS ESPECIALES EN VARIABLES + REGLAS LOGICAS -> NAn
# =============================================================================
vars_micro_nas = ["DPQ010", "SLQ120", "DBD900", "PAD680"] 

for col in vars_micro_nas:
    if col in datos.columns:
        # Si el valor es extremadamente cercano a 0 (menor que 0.01), lo forzamos a ser 0
        datos.loc[datos[col] < 0.01, col] = 0
        
# Códigos 7 (refused) y 9 (don't know) se convierten a Nan
vars_cuestionario = [
    "ALQ101", "PAQ650", "SLQ050", "SLQ120", "SMQ020",
    "DPQ010", "HSD010", "MCQ300C", "DMDEDUC2", "DIQ010"
]
for var in vars_cuestionario:
    if var in datos.columns:
        datos[var] = datos[var].replace({7: np.nan, 9: np.nan})

# Códigos de 4 dígitos: 7777, 9999, 5555, 6666 se convierten a Nan
if "DBD900" in datos.columns:
    datos["DBD900"] = datos["DBD900"].replace({
        7777: np.nan, # Refused
        9999: np.nan, # Don't know
        5555: 22      # "Más de 21 veces" -> Lo topamos en 22
    })
if "PAD680" in datos.columns:
    datos["PAD680"] = datos["PAD680"].replace({7777: np.nan, 9999: np.nan})
    
# Reglas logicas
# Sistólica siempre debe ser mayor que diastólica
mascara_inconsistente = (
    datos["BPXSY1"].notna() & datos["BPXDI1"].notna() &
    (datos["BPXSY1"] <= datos["BPXDI1"])
)
datos.loc[mascara_inconsistente, ["BPXSY1", "BPXDI1"]] = np.nan

# El Colesterol Total nunca puede ser menor que el HDL (Bueno)
inconsistencia_lipidos = (datos["LBXTC"].notna() & datos["LBDHDD"].notna()) & (datos["LBXTC"] < datos["LBDHDD"])
datos.loc[inconsistencia_lipidos, ["LBXTC", "LBDHDD"]] = np.nan

# Presión arterial (Límites máximos y mínimos biológicos)
datos["BPXSY1"]   = datos["BPXSY1"].where(datos["BPXSY1"].between(70, 300))
datos["BPXDI1"]   = datos["BPXDI1"].where(datos["BPXDI1"].between(1, 150))

# Antropometría
datos["BMXBMI"]    = datos["BMXBMI"].where(datos["BMXBMI"].between(10, 80))
datos["BMXWAIST"]  = datos["BMXWAIST"].where(datos["BMXWAIST"].between(40, 200))
    
# Variables de laboratorio 
datos["LBXTC"]     = datos["LBXTC"].where(datos["LBXTC"].between(50, 600))
datos["LBDHDD"]    = datos["LBDHDD"].where(datos["LBDHDD"].between(5, 150))
datos["LBXSATSI"]  = datos["LBXSATSI"].where(datos["LBXSATSI"].between(1, 1000))
datos["LBXSCR"]    = datos["LBXSCR"].where(datos["LBXSCR"].between(0.1, 15)) 
datos["LBXSUA"]    = datos["LBXSUA"].where(datos["LBXSUA"].between(1, 20))
datos["URDACT"]    = datos["URDACT"].where(datos["URDACT"].between(0, 10000))

datos["RIDAGEYR"]  = datos["RIDAGEYR"].where(datos["RIDAGEYR"].between(18, 120))
datos["INDFMPIR"]  = datos["INDFMPIR"].where(datos["INDFMPIR"].between(0, 5))

datos["RIAGENDR"]  = datos["RIAGENDR"].where(datos["RIAGENDR"].isin([1, 2]))
datos["ETNIA"]     = datos["ETNIA"].where(datos["ETNIA"].isin([1, 2, 3, 4, 6, 7]))
datos["DMDEDUC2"]  = datos["DMDEDUC2"].where(datos["DMDEDUC2"].isin([1, 2, 3, 4, 5]))
datos["ALQ101"]    = datos["ALQ101"].where(datos["ALQ101"].isin([1, 2]))
datos["PAQ650"]    = datos["PAQ650"].where(datos["PAQ650"].isin([1, 2]))
datos["SLQ050"]    = datos["SLQ050"].where(datos["SLQ050"].isin([1, 2]))
datos["SMQ020"]    = datos["SMQ020"].where(datos["SMQ020"].isin([1, 2]))
datos["MCQ300C"]   = datos["MCQ300C"].where(datos["MCQ300C"].isin([1, 2]))
datos["DPQ010"]    = datos["DPQ010"].where(datos["DPQ010"].isin([0, 1, 2, 3]))
datos["HSD010"]    = datos["HSD010"].where(datos["HSD010"].isin([1, 2, 3, 4, 5]))


# =============================================================================
# 6. CONSTRUCCIÓN DE LA VARIABLE RESPUESTA
# =============================================================================
# Un individuo se clasifica como DIABÉTICO si cumple al menos uno de:
#   - DIQ010 == 1  (diagnóstico médico declarado de diabetes)
#   - LBXGLU >= 126 mg/dL (glucosa en ayunas, criterio ADA)
#   - LBXGH >= 6.5 % (HbA1c, criterio ADA)
 
# Construimos primero la respuesta compuesta 
datos["DIABETES"] = (
    (datos["DIQ010"] == 1) |
    (datos["LBXGLU"] >= 126) |
    (datos["LBXGH"] >= 6.5)
).astype(int)

# Jerarquía diagnóstica: eliminamos los borderline SOLO si no superan los umbrales ADA.
# Los borderline confirmados por laboratorio se conservan como diabéticos (DIABETES=1).
n_antes = len(datos)
mascara_borderline_ambiguo = (datos["DIQ010"] == 3) & (datos["DIABETES"] == 0)
datos = datos[~mascara_borderline_ambiguo].copy()

print(f"Casos borderline ambiguos excluidos: {n_antes - len(datos)}\n")
print(f"Distribución final de la variable DIABETES:\n{datos['DIABETES'].value_counts()}\n")

# Eliminar filas donde los componentes analíticos sean NaN simultáneamente
sin_info = (
    datos["DIQ010"].isna() &
    datos["LBXGLU"].isna() &
    datos["LBXGH"].isna()
)
datos = datos[~sin_info].copy()

# Eliminar filas donde DIQ010 sea NaN 
datos = datos.dropna(subset=["DIQ010"]).copy()

print(datos["DIABETES"].value_counts())
print(f"Prevalencia: {datos['DIABETES'].mean():.1%}")

# Tabla de contingencia para DIABETES y DIQ010
print("--- COMPARATIVA: DIQ010 vs DIABETES ---")
print(pd.crosstab(
      datos["DIQ010"].map({1: "Diagnosticado",
                           2: "No diagnosticado",
                           3: "No diagnosticado"}),  
      datos["DIABETES"].map({0: "Sano", 1: "Diabético Real"}),
      margins=True, margins_name="Total"
))
# Vemos que hay 1437 (22% aprox) pacientes que no han sido diagnosticados
# nunca como diabéticos pero según sus índices de laboratorio sí lo son 

# Eliminar las variables de laboratorio usadas para construir la respuesta
datos = datos.drop(columns=["LBXGLU", "LBXGH"])
 

# =============================================================================
# 7. ANÁLISIS Y TRATAMIENTO DE NAs
# =============================================================================
# Funcion porcentaje de NAs
def porcentaje_nas(df, umbral=None):
    """Muestra el porcentaje de NAs de cada variable, ordenado de mayor a menor."""
    cols_excluir = ["SEQN", "DIABETES", "DIQ010"]
    df_pred = df.drop(columns=cols_excluir, errors="ignore")
    
    missing = (df_pred.isnull().mean() * 100).round(2).sort_values(ascending=False)
    missing = missing[missing > 0]
    
    if umbral is not None:
        missing = missing[missing > umbral]
    
    print(missing.to_string())
    print()
    return missing

print("=== Porcentaje de NAs ===")
porcentaje_nas(datos, 0)

# Eliminamos variables con mas de un 45% de NAs
vars_eliminar_cols = [ "LBDLDL", "LBXIN", "LBXTR", "HSD010", "SLQ120"]
datos = datos.drop(columns=vars_eliminar_cols)
porcentaje_nas(datos, 0)

# Eliminamos filas con NAs en variables con menos del 5% de missings :

# Eliminamos los NAs de PAQ650 (0.01%)
datos = datos.dropna(subset=["PAQ650"])
porcentaje_nas(datos)

# Eliminamos los NAs de SLQ050 (0.05%)
datos = datos.dropna(subset=["SLQ050"])
porcentaje_nas(datos)

# Eliminamos los NAs de PAD680 (0.6%)
datos = datos.dropna(subset=["PAD680"])
porcentaje_nas(datos)

# Eliminamos los NAs de SMQ020 (2.33%)
datos = datos.dropna(subset=["SMQ020"])
porcentaje_nas(datos)

# Eliminamos los NAs de DMDEDUC2 (2.89%)
datos = datos.dropna(subset=["DMDEDUC2"])
porcentaje_nas(datos)

# Eliminamos los NAs de MCQ300C (1.88%)
datos = datos.dropna(subset=["MCQ300C"])
porcentaje_nas(datos)

# Eliminamos los NAs de FSDHH (2.89%)
datos = datos.dropna(subset=["FSDHH"])
porcentaje_nas(datos)


# =============================================================================
# 8. REVISAR SI NAs SON ALEATORIOS O ESTRUCTURALES
# =============================================================================
# Comprobamos si el missing es estructural o aleatorio
# Mapa de ciclos por rango de SEQN
rangos_ciclo = {
    "E (2007-08)": (41475, 51623),
    "F (2009-10)": (51624, 62160),
    "G (2011-12)": (62161, 71916),
    "H (2013-14)": (73557, 83731),
    "I (2015-16)": (83732, 93702),
    "P (2017-20)": (109263, 124822),
}

def asignar_ciclo(seqn):
    for ciclo, (inicio, fin) in rangos_ciclo.items():
        if inicio <= seqn <= fin:
            return ciclo
    return "Desconocido"

for var in ["ALQ101", "DBD900", "DPQ010", "BPXDI1", "BPXSY1"]:
    print(f"\n=== Análisis de missing en {var} ===")
    datos["_missing"] = datos[var].isna().astype(int)
    datos["_ciclo"]   = datos["SEQN"].apply(asignar_ciclo)
    
    # 1. ¿Difiere el perfil demográfico?
    print("\nPerfil demográfico:")
    print(datos.groupby("_missing")[
        ["RIDAGEYR", "RIAGENDR", "BMXBMI", "INDFMPIR", "DIABETES"]
    ].mean().round(2))
    
    # 2. ¿Está concentrado en alguna etnia?
    print("\nDistribución por etnia:")
    print(pd.crosstab(datos["ETNIA"], datos["_missing"], normalize="index").round(2))
    
    # 3. ¿Está concentrado en algún ciclo?
    print("\nDistribución por ciclo:")
    print(pd.crosstab(datos["_ciclo"], datos["_missing"], normalize="index").round(2))
    
    datos = datos.drop(columns=["_missing", "_ciclo"])

# El ciclo P(2017-20) tiene 100% de missing, pues la pregunta sobre alcohol no se
# recogió en este ciclo. Eso es missing estructural, no aleatorio. Imputar un 40% 
# donde la mitad es estructural no tiene sentido, asi que elimino la variable.
# En el caso de DBD900 el missing es aleatorio, con lo que es imputable.
datos = datos.drop(columns=["ALQ101"])
porcentaje_nas(datos)


# =============================================================================
# 9. DIVISIÓN DEL DATASET (DOS FLUJOS INDEPENDIENTES)
# =============================================================================
# FLUJO 1: DIABETES COMPUESTA (Todos los adultos)
# Definimos vars predictoras (X) y vars objetivo (y), excluimos SEQN (identificador) del entrenamiento
X_global = datos.drop(columns=["SEQN", "DIABETES", "DIQ010"], errors="ignore")
y_global_cli = datos["DIABETES"]

X_train_cli, X_test_cli, y_train_cli, y_test_cli = train_test_split(
    X_global, y_global_cli, test_size=0.25, random_state=42, stratify=y_global_cli
)

# FLUJO 2: INFRADIAGNÓSTICO, solo entre positivos (DIABETES==1)
# Filtramos el dataset original para quedarnos SOLO con los diabéticos reales
datos_positivos = datos[datos["DIABETES"] == 1].copy()

# Infradiagnóstico: clase 0 = diagnosticado formalmente (DIQ010==1);
# clase 1 = no diagnosticado formalmente (DIQ010==2, o borderline rescatado DIQ010==3).
datos_positivos["INFRADIAGNOSTICO"] = datos_positivos["DIQ010"].isin([2, 3]).astype(int)

X_sub_global = datos_positivos.drop(columns=["SEQN", "DIABETES", "DIQ010", "INFRADIAGNOSTICO"], errors="ignore")
y_sub_global = datos_positivos["INFRADIAGNOSTICO"]

X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
    X_sub_global, y_sub_global, test_size=0.25, random_state=42, stratify=y_sub_global
)

print("=== Tamaños del Flujo 1 (Diabetes Compuesta) ===")
print(f"Train: {X_train_cli.shape} | Test: {X_test_cli.shape}")
print("\n=== Tamaños del Flujo 2 (Solo Casos Positivos) ===")
print(f"Train: {X_train_sub.shape} | Test: {X_test_sub.shape}")


# =============================================================================
# 10. IMPUTACIÓN DE NAs 
# =============================================================================
# Clasificamos nuestras columnas según su naturaleza para aplicar la regla correcta
# Variables numéricas
vars_num = [
    "RIDAGEYR", "INDFMPIR", "BPXSY1", "BPXDI1", "BMXBMI", "BMXWAIST", "LBXTC",
    "LBXSATSI", "LBDHDD", "URDACT", "LBXSCR", "LBXSUA", "DBD900", "PAD680"
]
# Variables categóricas
vars_cat = [
    "RIAGENDR", "ETNIA", "DMDEDUC2", "PAQ650", "SLQ050", "SMQ020",
    "MCQ300C", "DPQ010", "FSDHH"
]

valores_imputacion_cli = {col: X_train_cli[col].median() if col in vars_num else X_train_cli[col].mode()[0] for col in X_train_cli.columns}
for col in X_train_cli.columns:
    X_train_cli[col] = X_train_cli[col].fillna(valores_imputacion_cli[col])
    X_test_cli[col]  = X_test_cli[col].fillna(valores_imputacion_cli[col])

valores_imputacion_sub = {col: X_train_sub[col].median() if col in vars_num else X_train_sub[col].mode()[0] for col in X_train_sub.columns}
for col in X_train_sub.columns:
    X_train_sub[col] = X_train_sub[col].fillna(valores_imputacion_sub[col])
    X_test_sub[col]  = X_test_sub[col].fillna(valores_imputacion_sub[col])


# =============================================================================
# 11. TRATAMIENTO DE OUTLIERS Y CORRELACIONES (ADAPTADA)
# =============================================================================
# Asimetría (skewness) 
vars_num_cli = [c for c in vars_num if c in X_train_cli.columns]
asimetrias = X_train_cli[vars_num_cli].skew().sort_values(ascending=False)
print("=== Asimetría (skewness) de las variables numéricas ===")
print(asimetrias.round(2).to_string())

# Histogramas del Train (todas a la vez) para evaluar visualmente las distribuciones
X_train_cli[vars_num_cli].hist(figsize=(15, 12), bins=30)
plt.suptitle("Histogramas del Conjunto de Entrenamiento", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# Capping (percentiles 1 y 99), por flujo
for col in [c for c in vars_num if c in X_train_cli.columns]:
    inf_c, sup_c = X_train_cli[col].quantile(0.01), X_train_cli[col].quantile(0.99)
    X_train_cli[col], X_test_cli[col] = X_train_cli[col].clip(inf_c, sup_c), X_test_cli[col].clip(inf_c, sup_c)
    inf_s, sup_s = X_train_sub[col].quantile(0.01), X_train_sub[col].quantile(0.99)
    X_train_sub[col], X_test_sub[col] = X_train_sub[col].clip(inf_s, sup_s), X_test_sub[col].clip(inf_s, sup_s)


# Análisis de correlaciones (Spearman, robusto frente a la asimetría y los
# valores extremos). Umbral crítico de 0.80 para detectar pares redundantes.
vars_num_corr = [c for c in vars_num if c in X_train_cli.columns]
matriz_corr = X_train_cli[vars_num_corr].corr(method="spearman")
superior_triangulo = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
pares_altos = [
    (col, fila, superior_triangulo.loc[fila, col])
    for col in superior_triangulo.columns
    for fila in superior_triangulo.index
    if pd.notna(superior_triangulo.loc[fila, col]) and superior_triangulo.loc[fila, col] > 0.80
]
print("Pares con correlación crítica (Spearman > 0.80):")
for col, fila, val in pares_altos:
    print(f"- {col} vs {fila}: rho = {val:.2f}")
 
# Detectamos alta correlación entre el IMC y el perímetro de cintura -> eliminamos cintura.
X_train_cli = X_train_cli.drop(columns=["BMXWAIST"], errors="ignore")
X_test_cli  = X_test_cli.drop(columns=["BMXWAIST"], errors="ignore")
X_train_sub = X_train_sub.drop(columns=["BMXWAIST"], errors="ignore")
X_test_sub  = X_test_sub.drop(columns=["BMXWAIST"], errors="ignore")
 
if "BMXWAIST" in vars_num:
    vars_num.remove("BMXWAIST")


# =============================================================================
# 12. PREPARACIÓN GENERAL PARA LOS MODELOS
# =============================================================================
# Creamos un dataframe aparte con variables dummy para usar en regresión logística
# Añadimos drop_first=True que elimina una categoría por variable para evitar multicolinealidad perfecta.

def preparar_matrices_rl(X_train_df, X_test_df, categorias, numericas):
    X_tr_rl, X_te_rl = X_train_df.copy(), X_test_df.copy()
    for col in categorias:
        if col in X_tr_rl.columns:
            X_tr_rl[col], X_te_rl[col] = X_tr_rl[col].astype(str), X_te_rl[col].astype(str)
            
    X_tr_rl = pd.get_dummies(X_tr_rl, columns=[c for c in categorias if c in X_tr_rl.columns], drop_first=True)
    X_te_rl = pd.get_dummies(X_te_rl, columns=[c for c in categorias if c in X_te_rl.columns], drop_first=True)
    X_tr_rl, X_te_rl = X_tr_rl.align(X_te_rl, join="left", axis=1, fill_value=0)
    
    v_num = [c for c in numericas if c in X_tr_rl.columns]
    scaler = StandardScaler()
    X_tr_rl[v_num] = scaler.fit_transform(X_tr_rl[v_num])
    X_te_rl[v_num]  = scaler.transform(X_te_rl[v_num])
    return X_tr_rl, X_te_rl

# Generamos las versiones específicas escaladas/dummies para Regresión Logística
X_train_cli_rl, X_test_cli_rl = preparar_matrices_rl(X_train_cli, X_test_cli, vars_cat, vars_num)
X_train_sub_rl, X_test_sub_rl = preparar_matrices_rl(X_train_sub, X_test_sub, vars_cat, vars_num)


# =============================================================================
# 12b. SUBMUESTREO (UNDERSAMPLING) DEL TRAIN
# =============================================================================
# Se submuestrea por índice para que las versiones tree y rl conserven exactamente
# las mismas filas

FRAC_POS = 0.45   # objetivo de positivos en el train. 0.45 -> 45/55 


def submuestrear_train(X_tree, X_rl, y, frac_pos=0.45, random_state=42):
    """Undersampling aleatorio de la clase MAYORITARIA del train hasta alcanzar
    'frac_pos' de positivos. Solo reduce la mayoritaria (no duplica nada).
    Devuelve (X_tree_us, X_rl_us, y_us) con índices coherentes entre las tres."""
    rng = np.random.RandomState(random_state)
    idx_pos = y.index[y == 1]
    idx_neg = y.index[y == 0]
    n_pos, n_neg = len(idx_pos), len(idx_neg)

    if n_pos / (n_pos + n_neg) < frac_pos:          # positivos en minoría -> recortar negativos
        n_neg_obj = min(n_neg, int(round(n_pos * (1 - frac_pos) / frac_pos)))
        idx_neg = pd.Index(rng.choice(idx_neg.values, size=n_neg_obj, replace=False))
    else:                                            # positivos en mayoría -> recortar positivos
        n_pos_obj = min(n_pos, int(round(n_neg * frac_pos / (1 - frac_pos))))
        idx_pos = pd.Index(rng.choice(idx_pos.values, size=n_pos_obj, replace=False))

    idx_keep = pd.Index(rng.permutation(idx_pos.union(idx_neg).values))  # barajamos
    return X_tree.loc[idx_keep], X_rl.loc[idx_keep], y.loc[idx_keep]


# Apartamos primero un conjunto de VALIDACIÓN del train original (prevalencia REAL,
# NO se submuestrea y NO se usa para entrenar): sirve SOLO para elegir el umbral.
# El undersampling se aplica únicamente a la parte de ajuste. Así el
# umbral se calibra a la prevalencia real y el test sigue intacto en el cajón.
idx_fit_cli, idx_val_cli = train_test_split(
    y_train_cli.index, test_size=0.20, stratify=y_train_cli, random_state=42
)
idx_fit_sub, idx_val_sub = train_test_split(
    y_train_sub.index, test_size=0.20, stratify=y_train_sub, random_state=42
)

# Validación (prevalencia real, sin submuestrear)
X_val_cli, X_val_cli_rl, y_val_cli = (
    X_train_cli.loc[idx_val_cli], X_train_cli_rl.loc[idx_val_cli], y_train_cli.loc[idx_val_cli]
)
X_val_sub, X_val_sub_rl, y_val_sub = (
    X_train_sub.loc[idx_val_sub], X_train_sub_rl.loc[idx_val_sub], y_train_sub.loc[idx_val_sub]
)

# Undersampling
X_train_cli_us, X_train_cli_rl_us, y_train_cli_us = submuestrear_train(
    X_train_cli.loc[idx_fit_cli], X_train_cli_rl.loc[idx_fit_cli], y_train_cli.loc[idx_fit_cli],
    frac_pos=FRAC_POS
)
X_train_sub_us, X_train_sub_rl_us, y_train_sub_us = submuestrear_train(
    X_train_sub.loc[idx_fit_sub], X_train_sub_rl.loc[idx_fit_sub], y_train_sub.loc[idx_fit_sub],
    frac_pos=FRAC_POS
)

print("\n=== Undersampling + validación con prevalencia real (test intacto) ===")
for nombre, y_orig, y_us, y_val in [
    ("Flujo 1 (DIABETES)",   y_train_cli, y_train_cli_us, y_val_cli),
    ("Flujo 2 (Infradiag.)", y_train_sub, y_train_sub_us, y_val_sub),
]:
    print(f"{nombre}: train {len(y_orig):6d} (pos {y_orig.mean():.1%}) "
          f"-> fit submuestreado {len(y_us):6d} (pos {y_us.mean():.1%}) | "
          f"validación {len(y_val):5d} (pos {y_val.mean():.1%})")


# =============================================================================
# 12c. FUNCIONES DE EVALUACIÓN Y CV
# =============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def calcular_metricas(y_test, y_prob, y_pred):
    """Calcula las métricas de un clasificador binario y las devuelve en un dict."""
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    sensibilidad  = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision     = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1            = (2 * precision * sensibilidad / (precision + sensibilidad)
                     if (precision + sensibilidad) > 0 else np.nan)
    f2            = (5 * precision * sensibilidad / (4 * precision + sensibilidad)
                     if (4 * precision + sensibilidad) > 0 else np.nan)

    return {
        "exactitud":     (tp + tn) / (tp + tn + fp + fn),
        "sensibilidad":  sensibilidad,
        "especificidad": especificidad,
        "precision":     precision,
        "f1":            f1,
        "f2":            f2,
        "auc":           roc_auc_score(y_test, y_prob),
        "ap":            average_precision_score(y_test, y_prob),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def imprimir_metricas(nombre, m, labels=("Sano", "Diabético")):
    """labels = (clase_negativa, clase_positiva). La sensibilidad mide la
    clase positiva; la especificidad, la negativa."""
    neg, pos = labels
    print(f"\n{'='*55}")
    print(f"  {nombre}")
    print(f"{'='*55}")
    print(f"  Exactitud      : {m['exactitud']:.4f}")
    print(f"  Sensibilidad   : {m['sensibilidad']:.4f}   ← {pos} bien clasificados")
    print(f"  Especificidad  : {m['especificidad']:.4f}   ← {neg} bien clasificados")
    print(f"  Precisión      : {m['precision']:.4f}")
    print(f"  F1-Score       : {m['f1']:.4f}")
    print(f"  F2-Score       : {m['f2']:.4f}   ← prioriza recall (β=2)")
    print(f"  AUC-ROC        : {m['auc']:.4f}")
    print(f"  Avg Precision  : {m['ap']:.4f}")
    print(f"{'='*55}")
    print(f"  Matriz de confusión:")
    print(f"                       Pred {neg:<16} Pred {pos}")
    print(f"  Real {neg:<16} {m['tn']:7d}           {m['fp']:7d}")
    print(f"  Real {pos:<16} {m['fn']:7d}           {m['tp']:7d}")
    print(f"{'='*55}\n")


def evaluar_modelo(nombre, modelo, X_test, y_test, umbral=0.5, labels=("Sano", "Diabético")):
    """Evalúa un modelo: calcula métricas, las imprime y devuelve todo en un dict
    (incluye y_prob/y_pred, listo para plot_roc, plot_pr y tabla_resumen)."""
    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= umbral).astype(int)

    metricas = calcular_metricas(y_test, y_prob, y_pred)
    imprimir_metricas(nombre, metricas, labels=labels)

    return {"y_prob": y_prob, "y_pred": y_pred, **metricas}


def mejor_umbral_fbeta(y_true, y_prob, beta=2.0):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]          # el último punto (rec=0) no tiene umbral
    b2 = beta ** 2
    denom = b2 * prec + rec
    fbeta = np.where(denom > 0, (1 + b2) * prec * rec / denom, 0.0)
    if len(fbeta) == 0:
        return 0.5, np.nan
    i = int(np.argmax(fbeta))
    return float(thr[i]), float(fbeta[i])


def evaluar_con_umbral(nombre, modelo, X_val, y_val, X_test, y_test,
                       labels=("Sano", "Diabético"), beta=2.0):
    """Elige el umbral que maximiza F-beta en validación (prevalencia real, sin
    submuestrear) y evalúa el TEST con ese umbral fijo. El test no interviene en
    la elección del umbral, así que sigue siendo una estimación honesta."""
    p_val = modelo.predict_proba(X_val)[:, 1]
    umbral, fbeta_val = mejor_umbral_fbeta(y_val, p_val, beta=beta)
    print(f"  Umbral F{beta:g} óptimo (validación, prevalencia real): {umbral:.4f}"
          f"  |  F{beta:g} en validación: {fbeta_val:.4f}")
    resultado = evaluar_modelo(nombre, modelo, X_test, y_test, umbral=umbral, labels=labels)
    resultado["umbral"] = umbral
    return resultado


def plot_roc(resultados_dict, y_test, titulo):
    """Curva ROC para varios modelos sobre el mismo target."""
    plt.figure(figsize=(7, 5))
    for nombre, res in resultados_dict.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{nombre} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr(resultados_dict, y_test, titulo):
    """Curva Precision-Recall para varios modelos sobre el mismo target.
    La línea de base es la prevalencia (no 0.5, como en la ROC)."""
    prevalencia = y_test.mean()
    plt.figure(figsize=(7, 5))
    for nombre, res in resultados_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
        plt.plot(recall, precision, label=f"{nombre} (AP={res['ap']:.3f})")
    plt.axhline(prevalencia, color="k", linestyle="--", linewidth=0.8,
                label=f"Clasificador aleatorio (prevalencia={prevalencia:.3f})")
    plt.xlabel("Recall (Sensibilidad)")
    plt.ylabel("Precisión")
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()


def graficar_coeficientes(modelo, columnas, titulo, top_n=15, ax=None):
    """Grafica los top_n coeficientes (en valor absoluto) de un modelo de RL."""
    coefs = pd.Series(modelo.coef_[0], index=columnas)
    top = coefs.abs().sort_values(ascending=False).head(top_n).index
    coefs_top = coefs[top].sort_values()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    colores = ["firebrick" if v > 0 else "steelblue" for v in coefs_top]
    coefs_top.plot(kind="barh", color=colores, ax=ax)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Coeficiente (espacio logit)")
    return ax


def tabla_odds_ratios(modelo, columnas, top_n=15):
    """Tabla con coeficientes y odds ratios (OR = e^coef), ordenada por importancia.
    OR > 1 → aumenta la prob. de la clase positiva del flujo; OR < 1 → la reduce."""
    coefs = pd.Series(modelo.coef_[0], index=columnas)
    tabla = pd.DataFrame({"Coeficiente": coefs, "Odds Ratio": np.exp(coefs)})
    tabla = tabla.reindex(coefs.abs().sort_values(ascending=False).index)
    return tabla.head(top_n).round(4)


def graficar_importancias(modelo, columnas, titulo, top_n=15, ax=None, color="seagreen"):
    """Grafica las top_n variables más importantes según el modelo (MDI/Gini)."""
    importancias = pd.Series(modelo.feature_importances_, index=columnas)
    top = importancias.sort_values(ascending=False).head(top_n).sort_values()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    top.plot(kind="barh", color=color, ax=ax)
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel("Importancia")
    return ax


def tabla_resumen(resultados_dict):
    """Compila las métricas de varios modelos en un único DataFrame comparativo
    (útil para exportar directamente a LaTeX con .to_latex())."""
    filas = []
    for nombre, res in resultados_dict.items():
        filas.append({
            "Modelo": nombre,
            "Exactitud": res["exactitud"],
            "Sensibilidad": res["sensibilidad"],
            "Especificidad": res["especificidad"],
            "Precisión": res["precision"],
            "F1": res["f1"],
            "F2": res["f2"],
            "AUC-ROC": res["auc"],
            "Avg Precision": res["ap"],
        })
    return pd.DataFrame(filas).set_index("Modelo").round(4)


flujos = [
    {
        "key": "cli", "titulo": "DIABETES compuesto (undersampling)",
        "labels": ("Sano", "Diabético"), "class_weight": None,
        "X_train_rl": X_train_cli_rl_us, "X_test_rl": X_test_cli_rl,
        "X_train_tree": X_train_cli_us, "X_test_tree": X_test_cli,
        "X_val_rl": X_val_cli_rl, "X_val_tree": X_val_cli, "y_val": y_val_cli,
        "y_train": y_train_cli_us, "y_test": y_test_cli,
    },
    {
        "key": "sub", "titulo": "Infradiagnóstico (undersampling)",
        "labels": ("Diagnosticado", "Infradiagnosticado"), "class_weight": None,
        "X_train_rl": X_train_sub_rl_us, "X_test_rl": X_test_sub_rl,
        "X_train_tree": X_train_sub_us, "X_test_tree": X_test_sub,
        "X_val_rl": X_val_sub_rl, "X_val_tree": X_val_sub, "y_val": y_val_sub,
        "y_train": y_train_sub_us, "y_test": y_test_sub,
    },
]


# =============================================================================
# 13. REGRESIÓN LOGÍSTICA — BASE (sin regularización)
# =============================================================================
res_rl = {}

for f in flujos:
    modelo_instancia = LogisticRegression(
        penalty=None, solver="saga", max_iter=5000,
        random_state=42, class_weight=f["class_weight"],
    )
    modelo_instancia.fit(f["X_train_rl"], f["y_train"])

    resultado = evaluar_con_umbral(
        f"RL sin regularización — {f['titulo']}",
        modelo_instancia, f["X_val_rl"], f["y_val"], f["X_test_rl"], f["y_test"],
        labels=f["labels"], beta=2,
    )
    resultado["modelo"] = modelo_instancia
    res_rl[f["key"]] = resultado

# Matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, f in zip(axes.ravel(), flujos):
    ConfusionMatrixDisplay.from_predictions(
        f["y_test"], res_rl[f["key"]]["y_pred"],
        display_labels=list(f["labels"]), ax=ax,
    )
    ax.set_title(f"RL base | {f['titulo']}")
plt.tight_layout()
plt.show()


# =============================================================================
# 13.1 REGULARIZACIÓN: RIDGE, LASSO Y ELASTIC NET (GridSearchCV)
# =============================================================================
C_cli = np.logspace(-2, 1, 16)    
C_sub = np.logspace(-3, 0, 16)     

l1_cli = np.array([0.0, 0.1, 0.3, 0.5])
l1_sub = np.array([0.5, 0.7, 0.9, 1.0])

configuraciones_rl = {
    "Ridge (L2)":  {"penalty": "l2",
                    "param_grid": {"cli": {"C": C_cli}, "sub": {"C": C_sub}}},
    "Lasso (L1)":  {"penalty": "l1",
                    "param_grid": {"cli": {"C": C_cli}, "sub": {"C": C_sub}}},
    "Elastic Net": {"penalty": "elasticnet",
                    "param_grid": {"cli": {"C": C_cli, "l1_ratio": l1_cli},
                                   "sub": {"C": C_sub, "l1_ratio": l1_sub}}},
}

res_rl_variantes = {f["key"]: {} for f in flujos}

for f in flujos:
    res_rl_variantes[f["key"]]["Sin regularización"] = res_rl[f["key"]]

    for nombre_config, cfg in configuraciones_rl.items():
        base = LogisticRegression(
            penalty=cfg["penalty"],
            solver="saga",
            class_weight=f["class_weight"],
            max_iter=5000,
            random_state=42,
        )

        busqueda = GridSearchCV(
            base,
            param_grid=cfg["param_grid"][f["key"]],  
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        )
    
        busqueda.fit(f["X_train_rl"], f["y_train"])

        print(f"\n--- {nombre_config} — {f['titulo']} ---")
        print(f"  Mejores parámetros : {busqueda.best_params_}")
        print(f"  AUC en CV (Train)   : {busqueda.best_score_:.4f}")

        mejor_modelo = busqueda.best_estimator_
        resultado = evaluar_con_umbral(
            f"{nombre_config} — {f['titulo']}",
            mejor_modelo, f["X_val_rl"], f["y_val"], f["X_test_rl"], f["y_test"],
            labels=f["labels"], beta=2,
        )
        resultado["modelo"] = mejor_modelo
        res_rl_variantes[f["key"]][nombre_config] = resultado

for f in flujos:
    plot_roc(res_rl_variantes[f["key"]], f["y_test"], f"Curva ROC — {f['titulo']}")
    plot_pr(res_rl_variantes[f["key"]], f["y_test"], f"Curva Precision-Recall — {f['titulo']}")


# =============================================================================
# 13.4 COEFICIENTES Y ODDS RATIOS — PARA CADA VARIANTE
# =============================================================================
for f in flujos:
    for nombre_config, resultado in res_rl_variantes[f["key"]].items():
        modelo = resultado["modelo"]

        fig, ax = plt.subplots(figsize=(7, 6))
        graficar_coeficientes(modelo, f["X_train_rl"].columns,
                               f"{nombre_config} — {f['titulo']}", ax=ax)
        plt.tight_layout()
        plt.show()

        print(f"\n=== Top variables — {nombre_config} — {f['titulo']} ===")
        print(tabla_odds_ratios(modelo, f["X_train_rl"].columns))


# =============================================================================
# 13.5 SELECCIÓN DE VARIABLES: COEFICIENTES QUE LASSO/ELASTIC NET LLEVAN A 0
# =============================================================================
for f in flujos:
    print(f"\n=== Variables eliminadas (coeficiente = 0) — {f['titulo']} ===")
    for nombre_config in ["Lasso (L1)", "Elastic Net"]:
        modelo = res_rl_variantes[f["key"]][nombre_config]["modelo"]
        coefs = pd.Series(modelo.coef_[0], index=f["X_train_rl"].columns)
        eliminadas = coefs[coefs == 0].index.tolist()
        print(f"{nombre_config}: {len(eliminadas)} de {len(coefs)} variables eliminadas")
        if eliminadas:
            print(f"  → {eliminadas}")


# =============================================================================
# 13.6 TABLA RESUMEN COMPARATIVA
# =============================================================================
for f in flujos:
    print(f"\n=== Resumen — {f['titulo']} ===")
    print(tabla_resumen(res_rl_variantes[f["key"]]))


# =============================================================================
# 14. ÁRBOLES DE DECISIÓN
# =============================================================================

# -----------------------------------------------------------------------------
# 14.1 Árbol sin restricciones — para ilustrar el sobreajuste
# -----------------------------------------------------------------------------
for f in flujos:
    arbol_libre = DecisionTreeClassifier(random_state=42, class_weight=f["class_weight"])
    arbol_libre.fit(f["X_train_tree"], f["y_train"])

    auc_train = roc_auc_score(f["y_train"], arbol_libre.predict_proba(f["X_train_tree"])[:, 1])
    auc_test  = roc_auc_score(f["y_test"],  arbol_libre.predict_proba(f["X_test_tree"])[:, 1])
    print(f"\n--- Árbol sin restricciones — {f['titulo']} ---")
    print(f"  Profundidad alcanzada : {arbol_libre.get_depth()}")
    print(f"  Nº de hojas           : {arbol_libre.get_n_leaves()}")
    print(f"  AUC en Train          : {auc_train:.4f}")
    print(f"  AUC en Test           : {auc_test:.4f}   ← la caída indica sobreajuste")

# -----------------------------------------------------------------------------
# 14.2 Búsqueda de hiperparámetros (GridSearchCV)
# -----------------------------------------------------------------------------
param_grid_arbol = {
    "max_depth":         [3, 5, 7, 10, 15, None],
    "min_samples_leaf":  [1, 5, 10, 20, 50],
    "min_samples_split": [2, 10, 20],
}

res_arbol = {f["key"]: {} for f in flujos}

for f in flujos:
    busqueda = GridSearchCV(
        DecisionTreeClassifier(random_state=42, class_weight=f["class_weight"]),
        param_grid=param_grid_arbol,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    busqueda.fit(f["X_train_tree"], f["y_train"])

    print(f"\n--- Mejor árbol — {f['titulo']} ---")
    print(f"  Mejores parámetros : {busqueda.best_params_}")
    print(f"  AUC en CV (Train)   : {busqueda.best_score_:.4f}")

    mejor_arbol = busqueda.best_estimator_
    resultado = evaluar_con_umbral(
        f"Árbol de Decisión — {f['titulo']}",
        mejor_arbol, f["X_val_tree"], f["y_val"], f["X_test_tree"], f["y_test"],
        labels=f["labels"], beta=2,
    )
    resultado["modelo"] = mejor_arbol
    res_arbol[f["key"]]["Árbol de Decisión"] = resultado

# -----------------------------------------------------------------------------
# 14.3 Visualización del árbol (ilustrativa)
# -----------------------------------------------------------------------------
for f in flujos:
    arbol_visual = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight=f["class_weight"])
    arbol_visual.fit(f["X_train_tree"], f["y_train"])

    plt.figure(figsize=(18, 8))
    plot_tree(
        arbol_visual, feature_names=f["X_train_tree"].columns,
        class_names=list(f["labels"]),
        filled=True, rounded=True, fontsize=8,
    )
    plt.title(f"Árbol de Decisión (profundidad 3, solo ilustrativo) — {f['titulo']}")
    plt.show()

# -----------------------------------------------------------------------------
# 14.4 Importancia de variables
# -----------------------------------------------------------------------------
for f in flujos:
    modelo = res_arbol[f["key"]]["Árbol de Decisión"]["modelo"]
    fig, ax = plt.subplots(figsize=(7, 6))
    graficar_importancias(modelo, f["X_train_tree"].columns,
                           f"Árbol de Decisión — {f['titulo']}", ax=ax)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 15. RANDOM FOREST
# =============================================================================
param_grid_rf_cli = {
    "n_estimators":     [700, 1000, 1500], 
    "max_depth":        [15, 20, 30, None],
    "min_samples_leaf": [5, 10, 20],
    "max_features":     ["sqrt", 0.3],     
}
param_grid_rf_sub = {
    "n_estimators":     [200, 400, 600],
    "max_depth":        [4, 6, 8, 10],       
    "min_samples_leaf": [10, 20, 40],
    "max_features":     ["sqrt", "log2"],
}
grids_rf = {"cli": param_grid_rf_cli, "sub": param_grid_rf_sub}

res_rf = {f["key"]: {} for f in flujos}
for f in flujos:
    rf_cw = "balanced_subsample" if f["class_weight"] else None
    busqueda_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight=rf_cw),
        param_grid=grids_rf[f["key"]],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    busqueda_rf.fit(f["X_train_tree"], f["y_train"])

    print(f"\n--- Mejor Random Forest — {f['titulo']} ---")
    print(f"  Mejores parámetros : {busqueda_rf.best_params_}")
    print(f"  AUC en CV (Train)   : {busqueda_rf.best_score_:.4f}")

    mejor_rf = busqueda_rf.best_estimator_
    resultado = evaluar_con_umbral(
        f"Random Forest — {f['titulo']}",
        mejor_rf, f["X_val_tree"], f["y_val"], f["X_test_tree"], f["y_test"],
        labels=f["labels"], beta=2,
    )
    resultado["modelo"] = mejor_rf
    res_rf[f["key"]]["Random Forest"] = resultado
    

# -----------------------------------------------------------------------------
# 15.2 Importancia de variables: MDI vs Permutación
# -----------------------------------------------------------------------------
for f in flujos:
    modelo = res_rf[f["key"]]["Random Forest"]["modelo"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    graficar_importancias(modelo, f["X_train_tree"].columns,
                           f"Importancia MDI — {f['titulo']}", ax=axes[0])

    perm = permutation_importance(
        modelo, f["X_test_tree"], f["y_test"], scoring="roc_auc",
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    importancias_perm = pd.Series(perm.importances_mean, index=f["X_train_tree"].columns)
    top_perm = importancias_perm.sort_values(ascending=False).head(15).sort_values()
    top_perm.plot(kind="barh", color="darkorange", ax=axes[1])
    axes[1].set_title(f"Importancia por Permutación — {f['titulo']}", fontsize=11)
    axes[1].set_xlabel("Caída media de AUC al permutar")

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 15.3 Comparativa: Árbol de Decisión vs Random Forest
# -----------------------------------------------------------------------------
for f in flujos:
    comparacion = {
        "Árbol de Decisión": res_arbol[f["key"]]["Árbol de Decisión"],
        "Random Forest":     res_rf[f["key"]]["Random Forest"],
    }
    plot_roc(comparacion, f["y_test"], f"Árbol vs Random Forest — ROC — {f['titulo']}")
    plot_pr(comparacion, f["y_test"], f"Árbol vs Random Forest — PR — {f['titulo']}")

    print(f"\n=== Resumen — {f['titulo']} ===")
    print(tabla_resumen(comparacion))


# =============================================================================
# 16. STOCHASTIC GRADIENT BOOSTING (SGB)
# =============================================================================
param_grid_gb_cli = {                        
    "n_estimators":     [600, 1000, 1500],   
    "learning_rate":    [0.02, 0.05, 0.1],
    "max_depth":        [2, 3, 4],          
    "subsample":        [0.8, 1.0],          
    "min_samples_leaf": [20, 50, 100],       
}
param_grid_gb_sub = {                        
    "n_estimators":     [400, 800, 1500],    
    "learning_rate":    [0.005, 0.01, 0.03], 
    "max_depth":        [3, 4, 5],
    "subsample":        [0.5, 0.7],          
    "min_samples_leaf": [50, 100],           
}
grids_gb = {"cli": param_grid_gb_cli, "sub": param_grid_gb_sub}

res_gb = {f["key"]: {} for f in flujos}

for f in flujos:
    busqueda_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid=grids_gb[f["key"]],
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    busqueda_gb.fit(f["X_train_tree"], f["y_train"])

    print(f"\n--- Mejor Gradient Boosting — {f['titulo']} ---")
    print(f"  Mejores parámetros : {busqueda_gb.best_params_}")
    print(f"  AUC en CV (Train)   : {busqueda_gb.best_score_:.4f}")

    mejor_gb = busqueda_gb.best_estimator_
    resultado = evaluar_con_umbral(
        f"Gradient Boosting — {f['titulo']}",
        mejor_gb, f["X_val_tree"], f["y_val"], f["X_test_tree"], f["y_test"],
        labels=f["labels"], beta=2,
    )
    resultado["modelo"] = mejor_gb
    res_gb[f["key"]]["Gradient Boosting"] = resultado

# -----------------------------------------------------------------------------
# 16.1 Importancia de variables: MDI vs Permutación
# -----------------------------------------------------------------------------
for f in flujos:
    modelo = res_gb[f["key"]]["Gradient Boosting"]["modelo"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    graficar_importancias(modelo, f["X_train_tree"].columns,
                           f"Importancia MDI — {f['titulo']}", ax=axes[0], color="darkviolet")

    perm = permutation_importance(
        modelo, f["X_test_tree"], f["y_test"], scoring="roc_auc",
        n_repeats=10, random_state=42, n_jobs=-1,
    )
    importancias_perm = pd.Series(perm.importances_mean, index=f["X_train_tree"].columns)
    top_perm = importancias_perm.sort_values(ascending=False).head(15).sort_values()
    top_perm.plot(kind="barh", color="darkorange", ax=axes[1])
    axes[1].set_title(f"Importancia por Permutación — {f['titulo']}", fontsize=11)
    axes[1].set_xlabel("Caída media de AUC al permutar")

    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# 16.2 Comparativa final: Árbol vs Random Forest vs Gradient Boosting
# -----------------------------------------------------------------------------
for f in flujos:
    comparacion = {
        "Árbol de Decisión": res_arbol[f["key"]]["Árbol de Decisión"],
        "Random Forest":     res_rf[f["key"]]["Random Forest"],
        "Gradient Boosting": res_gb[f["key"]]["Gradient Boosting"],
    }
    plot_roc(comparacion, f["y_test"], f"Comparativa modelos basados en árboles — ROC — {f['titulo']}")
    plot_pr(comparacion, f["y_test"], f"Comparativa modelos basados en árboles — PR — {f['titulo']}")

    print(f"\n=== Resumen — {f['titulo']} ===")
    print(tabla_resumen(comparacion))

