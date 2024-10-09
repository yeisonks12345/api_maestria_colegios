from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import f1_score, recall_score

df = pd.read_csv('df_output/imputa_knn/imputa_knn.csv')


x =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 
y =df['GLOBAL_CATEGORICO']

X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

xgb_classifier = xgb.XGBClassifier(n_estimators=2200, 
                                   objective="multi:softprob", 
                                   tree_method='hist', 
                                   eta=0.1, 
                                   max_depth=3,
                                   colsample_bytree= 0.8,
                                   learning_rate= 0.01,
                                   subsample= 0.7, 
                                   enable_categorical=True)

xgb_classifier.fit(X_train, y_train)


predictions = xgb_classifier.predict(X_test)

import pickle
#pickle.dump(xgb_classifier,open('icfes_clasi.pkl','wb'))

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
#importance_df.to_csv('df_output/feature_importance_xgb.csv',index=False)

print('accuracy: ',accuracy_score(y_test, predictions))
print('f1_score: ',f1_score(y_test, predictions, average='weighted'))  # Para manejo de multiclase
print('recall: ',recall_score(y_test, predictions, average='weighted'))

#Acuracy 0.654657, f1_score 0.654397, recall 0.654657