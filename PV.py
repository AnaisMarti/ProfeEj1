import pandas as pd

# Cargar el conjunto de datos
data = pd.read_csv('winequality-red.csv')
# Mostrar las primeras filas
print(data.head())

# Estadísticas descriptivas
print(data.describe())

# Distribución de la variable objetivo
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(data['quality'])
plt.title('Distribución de la Calidad del Vino')
plt.xlabel('Calidad')
plt.ylabel('Frecuencia')
plt.show()
# Gráficos de dispersión de algunas variables
sns.scatterplot(x='alcohol', y='quality', data=data)
plt.title('Relación entre Alcohol y Calidad')
plt.show()

sns.boxplot(x='quality', y='volatile acidity', data=data)
plt.title('Acidez Volátil según Calidad')
plt.show()
X = data.drop('quality', axis=1)
y = data['quality']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
accuracy_tree = accuracy_score(y_test, y_pred_tree.round())

print(f'Error Cuadrático Medio (Árbol de Decisión): {mse_tree}')
print(f'Precisión (Árbol de Decisión): {accuracy_tree}')
from sklearn.tree import export_text

print(export_text(tree_model, feature_names=list(data.columns[:-1])))
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf.round())

print(f'Error Cuadrático Medio (Random Forest): {mse_rf}')
print(f'Precisión (Random Forest): {accuracy_rf}')
print(f'Comparación de Resultados:')
print(f'Árbol de Decisión - MSE: {mse_tree}, Precisión: {accuracy_tree}')
print(f'Random Forest - MSE: {mse_rf}, Precisión: {accuracy_rf}')
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print(f'Mejores parámetros: {grid_search.best_params_}')
importances = rf_model.feature_importances_

# Crear un DataFrame para visualizar
feature_importance_df = pd.DataFrame({
    'feature': data.columns[:-1],
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feature_importance_df)
sns.barplot(x='importance', y='feature', data=feature_importance_df.head(10))
plt.title('Características más Importantes para Predecir la Calidad del Vino')
plt.show()
