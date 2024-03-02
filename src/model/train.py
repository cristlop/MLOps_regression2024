import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn import datasets, cluster
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Paso 0: Instalar W&B
!pip instalar wandb -qU

# Paso 1: Importar W&B e iniciar sesión
import wandb
wandb.login()

# Regresión

# Echamos un vistazo a un ejemplo rápido
# Cargar datos
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X, y = X[::2], y[::2]  # submuestra para una demostración más rápida
wandb.errors.term._show_warnings = False
# ignorar las advertencias sobre los gráficos que se están construyendo a partir de un subconjunto de datos

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Modelo de tren, obtén predicciones
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Paso 2: Inicializar la ejecución de W&B
run = wandb.init(project='MLOps_regression2024', name="regression")

# Paso 3: Visualizar el rendimiento del modelo

# Parcela residual
# Mide y traza los valores objetivo previstos (eje y) frente a la diferencia entre los valores objetivo reales y previstos (eje x), 
# así como la distribución del error residual.
wandb.sklearn.plot_residuals(reg, X_train, y_train)

# Candidato a atípico
# Mide la influencia de un punto de datos en el modelo de regresión a través de la distancia de Cook.
# Los casos con influencias muy sesgadas podrían ser potencialmente valores atípicos. Útil para la detección de valores atípicos.
wandb.sklearn.plot_outlier_candidates(reg, X_train, y_train)

# Todo en uno: trama de regresión
# Usando esto todo en una API, se puede:
# Registro de resumen de métricas
# Curva de aprendizaje de registro
# Registrar candidatos atípicos
# Registro de parcela residual
wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')

wandb.finish()
