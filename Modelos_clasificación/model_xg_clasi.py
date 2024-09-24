from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb

df = pd.read_csv('df_output/balanceado_sinfaltantes.csv')

"""Se eliminan todas las variables que tengan relacion con el icfes, o datos que se 
obtienen unicamente despues de que el estudiante presente la prueba del icfes, """

x =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL','INSE','ESTU_CONSECUTIVO','ESTU_NSE_ESTABLECIMIENTO','ESTU_NSE_ESTABLECIMIENTO','COLE_CODIGO_ICFES','COLE_DEPTO_UBICACION','COLE_COD_DEPTO_UBICACION','COLE_MCPIO_UBICACION','ESTU_ESTUDIANTE','COLE_COD_MCPIO_UBICACION','ESTU_PRIVADO_LIBERTAD','ESTU_COD_MCPIO_PRESENTACION','ESTU_ESTADOINVESTIGACION','ESTU_MCPIO_PRESENTACION','ESTU_DEPTO_PRESENTACION','ESTU_COD_DEPTO_PRESENTACION','ESTU_COD_RESIDE_DEPTO','ESTU_DEPTO_RESIDE'],axis=1) 
y =df['GLOBAL_CATEGORICO']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.2,random_state=2)

xgb_classifier = xgb.XGBClassifier(n_estimators=2200, 
                                   objective="multi:softprob", 
                                   tree_method='hist', 
                                   eta=0.1, 
                                   max_depth=3,
                                   colsample_bytree= 0.8,
                                   learning_rate= 0.01,
                                   subsample= 0.7, 
                                   enable_categorical=True)

xgb_classifier.fit(Xtrain, Ytrain)


preds = xgb_classifier.predict(Xtest)
print(accuracy_score(Ytest, preds))
import pickle
#pickle.dump(xgb_classifier,open('icfes_clasi.pkl','wb'))

#print(len(x.columns))
#el modelo genera un accuracy de 0.63 sin grid search, con grid search sube a 0.6477 existen 43 variables predictoras.

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
grid_search.fit(Xtrain, Ytrain)
print(grid_search.best_params_)
"""

#Imprimir las variables principales de xgboost
importances = xgb_classifier.feature_importances_
feature_names = Xtrain.columns
# Crear un DataFrame con los nombres de las características y sus importancias
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Ordenar las características por importancia (de mayor a menor)
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Imprimir el DataFrame
#print(importance_df.head(20))
importance_df.to_csv('df_output/feature_importance_xgb.csv',index=False)