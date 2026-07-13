# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import json, gzip, pickle, os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

CATEGORICAS = ["SEX", "EDUCATION", "MARRIAGE"]
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_TRAIN = os.path.join(RAIZ, "files", "input", "train_data.csv.zip")
RUTA_TEST = os.path.join(RAIZ, "files", "input", "test_data.csv.zip")
RUTA_MODELO = os.path.join(RAIZ, "files", "models", "model.pkl.gz")
RUTA_METRICAS = os.path.join(RAIZ, "files", "output", "metrics.json")


def limpiar(df):
    """Limpia el dataset segun las reglas del enunciado."""
    df = df.rename(columns={"default payment next month": "default"}).drop(columns=["ID"]).dropna()
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v > 4 else v)
    return df


def cargar_datos():
    """Carga y limpia train y test."""
    train = limpiar(pd.read_csv(RUTA_TRAIN, index_col=False, compression="zip"))
    test = limpiar(pd.read_csv(RUTA_TEST, index_col=False, compression="zip"))
    return train.drop(columns=["default"]), train["default"], test.drop(columns=["default"]), test["default"]


def crear_modelo(x_train, y_train):
    """Construye el pipeline y optimiza sus hiperparametros con GridSearchCV."""
    numericas = [c for c in x_train.columns if c not in CATEGORICAS]
    pipeline = Pipeline([
        ("preprocesador", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
            ("num", MinMaxScaler(), numericas),
        ])),
        ("selector", SelectKBest(score_func=f_classif)),
        ("clasificador", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    grilla = {
        "selector__k": list(range(1, 24)),
        "clasificador__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "clasificador__solver": ["liblinear"],
    }
    modelo = GridSearchCV(pipeline, grilla, cv=10, scoring="balanced_accuracy", n_jobs=-1, refit=True)
    modelo.fit(x_train, y_train)
    return modelo


def guardar_modelo(modelo):
    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    with gzip.open(RUTA_MODELO, "wb") as archivo:
        pickle.dump(modelo, archivo)


def metricas(nombre, y_real, y_pred):
    return {
        "type": "metrics", "dataset": nombre,
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, y_pred),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }


def matriz_confusion(nombre, y_real, y_pred):
    m = confusion_matrix(y_real, y_pred)
    return {
        "type": "cm_matrix", "dataset": nombre,
        "true_0": {"predicted_0": int(m[0][0]), "predicted_1": int(m[0][1])},
        "true_1": {"predicted_0": int(m[1][0]), "predicted_1": int(m[1][1])},
    }


def guardar_metricas(modelo, x_train, y_train, x_test, y_test):
    """Calcula y guarda metricas y matrices de confusion en metrics.json."""
    os.makedirs(os.path.dirname(RUTA_METRICAS), exist_ok=True)
    yt_pred, ye_pred = modelo.predict(x_train), modelo.predict(x_test)
    filas = [
        metricas("train", y_train, yt_pred), metricas("test", y_test, ye_pred),
        matriz_confusion("train", y_train, yt_pred), matriz_confusion("test", y_test, ye_pred),
    ]
    with open(RUTA_METRICAS, "w") as archivo:
        archivo.write("\n".join(json.dumps(fila) for fila in filas) + "\n")


def main():
    x_train, y_train, x_test, y_test = cargar_datos()
    modelo = crear_modelo(x_train, y_train)
    guardar_modelo(modelo)
    guardar_metricas(modelo, x_train, y_train, x_test, y_test)


main()