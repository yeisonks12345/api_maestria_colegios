from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('df_output/balanceado_sinfaltantes.csv')

"""Se eliminan todas las variables que tengan relacion con el icfes, o datos que se 
obtienen unicamente despues de que el estudiante presente la prueba del icfes"""

x =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL','INSE','ESTU_CONSECUTIVO','ESTU_NSE_ESTABLECIMIENTO','ESTU_NSE_ESTABLECIMIENTO','COLE_CODIGO_ICFES','COLE_DEPTO_UBICACION','COLE_COD_DEPTO_UBICACION','COLE_MCPIO_UBICACION','ESTU_ESTUDIANTE','COLE_COD_MCPIO_UBICACION','ESTU_PRIVADO_LIBERTAD','ESTU_COD_MCPIO_PRESENTACION','ESTU_ESTADOINVESTIGACION','ESTU_MCPIO_PRESENTACION','ESTU_DEPTO_PRESENTACION','ESTU_COD_DEPTO_PRESENTACION','ESTU_COD_RESIDE_DEPTO','ESTU_DEPTO_RESIDE'],axis=1) 
y =df['GLOBAL_CATEGORICO']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(x,y,test_size=0.2,random_state=2)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(n_estimators=2200, objective="multi:softprob", tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)
xgb_classifier.fit(Xtrain, Ytrain)

from sklearn.metrics import accuracy_score
preds = xgb_classifier.predict(Xtest)
#print(accuracy_score(Ytest, preds))
import pickle
#pickle.dump(xgb_classifier,open('icfes_clasi.pkl','wb'))

#print(len(x.columns))
#el modelo genera un accuracy de 0.63, existen 43 variables predictoras.


