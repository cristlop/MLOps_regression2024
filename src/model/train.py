import warnings
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

# Importar W&B e iniciar sesión
import wandb
wandb.login()

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# REGRESION

# Cargar datos
housing = datasets.fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X, y = X[::2], y[::2]  # submuestra para una demostración más rápida

# ignorar las advertencias sobre los gráficos que se están construyendo a partir de un subconjunto de datos
wandb.errors.term._show_warnings = False

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Modelo de tren, obtén predicciones
reg = Ridge()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Inicializar la ejecución de W&B
run = wandb.init(project='MLOps_regression2024', name="regression")

# Visualizar el rendimiento del modelo

# Curva de Aprendizaje
wandb.sklearn.plot_learning_curve(reg, X_train, y_train)

# Parcela Residual
wandb.sklearn.plot_residuals(reg, X_train, y_train)

# Obtener predicciones en el conjunto de prueba
y_pred = reg.predict(X_test)

# Calcular métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Imprimir métricas
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R2 Score: {r2}")

wandb.finish()
