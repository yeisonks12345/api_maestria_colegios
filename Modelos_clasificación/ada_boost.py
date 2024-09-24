from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('df_output/balanceado_sinfaltantes.csv')
# Dividir los datos
X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL','INSE','ESTU_CONSECUTIVO','ESTU_NSE_ESTABLECIMIENTO','ESTU_NSE_ESTABLECIMIENTO','COLE_CODIGO_ICFES','COLE_DEPTO_UBICACION','COLE_COD_DEPTO_UBICACION','COLE_MCPIO_UBICACION','ESTU_ESTUDIANTE','COLE_COD_MCPIO_UBICACION','ESTU_PRIVADO_LIBERTAD','ESTU_COD_MCPIO_PRESENTACION','ESTU_ESTADOINVESTIGACION','ESTU_MCPIO_PRESENTACION','ESTU_DEPTO_PRESENTACION','ESTU_COD_DEPTO_PRESENTACION','ESTU_COD_RESIDE_DEPTO','ESTU_DEPTO_RESIDE'],axis=1) 
y =df['GLOBAL_CATEGORICO']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=2)

ada = AdaBoostClassifier(n_estimators=100)
ada.fit(Xtrain, Ytrain)
predictions = ada.predict(Xtest)
print("Accuracy (AdaBoost):", accuracy_score(Ytest, predictions))

#accuracy 0.631966