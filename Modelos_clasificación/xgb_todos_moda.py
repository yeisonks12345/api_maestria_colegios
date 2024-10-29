from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import f1_score, recall_score
import numpy as np

df = pd.read_csv('df_output/imputacion_todos_moda/imputacion_todos_moda.csv')
#con base en feature importance se prueban con diferente cantidad de variables para el despliegue, se borrar las columnas:
#borrar = ['GLOBAL_CATEGORICO','PUNT_GLOBAL','COLE_AREA_UBICACION','COLE_NOMBRE_ESTABLECIMIENTO','COLE_SEDE_PRINCIPAL','COLE_NOMBRE_SEDE','COLE_GENERO','COLE_NATURALEZA','ESTU_TIPOREMUNERACION','ESTU_DEDICACIONINTERNET','FAMI_PERSONASHOGAR','COLE_COD_DANE_ESTABLECIMIENTO','FAMI_TRABAJOLABORPADRE','FAMI_CUARTOSHOGAR','COLE_COD_DANE_SEDE','FAMI_TIENESERVICIOTV','FAMI_COMECEREALFRUTOSLEGUMBRE','FAMI_TRABAJOLABORMADRE','PERIODO','FAMI_TIENECONSOLAVIDEOJUEGOS','FAMI_TIENELAVADORA','FAMI_TIENEHORNOMICROOGAS','ESTU_PAIS_RESIDE','ESTU_MCPIO_RESIDE','ESTU_NACIONALIDAD','ESTU_DEPTO_RESIDE','COLE_MCPIO_UBICACION','COLE_DEPTO_UBICACION']
borrar = ['GLOBAL_CATEGORICO','PUNT_GLOBAL']
x =df.drop(borrar,axis=1) 
y =df['GLOBAL_CATEGORICO']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
# Crear el clasificador XGBoost multiclase
xgb_classifier = xgb.XGBClassifier(
    n_estimators=2200, 
    objective="multi:softprob", 
    tree_method='hist', 
    eta=0.1, 
    max_depth=3,
    colsample_bytree=0.8,
    learning_rate=0.01,
    subsample=0.7, 
    enable_categorical=True,
    
)
xgb_classifier.fit(X_train, y_train, sample_weight=sample_weights)
predictions = xgb_classifier.predict(X_test)

y_train_pred = xgb_classifier.predict(X_train)

"""Uso de grid search para encontrar los mejores parametros
{'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 1500, 'subsample': 0.7}

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [500, 1000, 1500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1],
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective="multi:softprob", tree_method='hist', enable_categorical=True), 
                           param_grid=param_grid, 
                           scoring='accuracy', 
                           cv=3, 
                           verbose=1)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
"""

#Imprimir las variables principales de xgboost
importances = xgb_classifier.feature_importances_
feature_names = X_train.columns
# Crear un DataFrame con los nombres de las características y sus importancias
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Ordenar las características por importancia (de mayor a menor)
#importance_df = importance_df.sort_values(by='Importance', ascending=False)
#Imprimir el DataFrame
#print(importance_df.head(20))
#importance_df.to_csv('df_output/borrando_faltantes/feature_importance_xgb.csv',index=False)

print('accuracy: ',accuracy_score(y_test, predictions))
print('f1_score: ',f1_score(y_test, predictions, average='weighted'))  # Para manejo de multiclase
print('recall: ',recall_score(y_test, predictions, average='weighted'))

#Se usa el mejor modelo para la app
#import pickle
#pickle.dump(xgb_classifier,open('icfes_clasi.pkl','wb'))


print('accuracy_train: ',accuracy_score(y_train, y_train_pred))

# el modelo xg boost con todos los datos 137000 registros, con el metodo de imputacion de la moda y 
# sin balance genera accuracy: 0.734736, f1score: 0.716188, recall: 0.734736
# con el train accuracy 0.7422, no se evidencia overfiting

#al balancear en xgboost genera un accuracy 0.6626, f1score 0.6802, recall 0.6626, train 0.66