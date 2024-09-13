#cambiar el enfoque y validar con puntaje global con rangos <=279 insufi, 280 y 359 satis, 360 avanz
#se seleccionan 6 variables con fscore arrojann accuracy 0.72 en colab esta el modelo, el INSE resume variables de nivel socioeconomico
#https://colab.research.google.com/drive/1ZStw9hLVrVQZqSHZx_O--PwJCVBkF_96#scrollTo=OED3_bE0kCx_
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('df_output\datos_codifcadobalanceados.csv')
variables_principales= ['INSE','EDAD'
                        ,'ESTU_DEDICACIONLECTURADIARIA','ESTU_DEDICACIONINTERNET','COLE_JORNADA','COLE_COD_DANE_ESTABLECIMIENTO']
df_y = df[['PUNT_GLOBAL']]
def y_cambio(i):
    if i<=279:
        return 0
    elif i >=280 and i <= 359:
        return 1
 
    
    elif i >=360:
        return 2

df_y['PUNT_GLOBAL'] = df_y['PUNT_GLOBAL'].apply(y_cambio)
Y= df_y['PUNT_GLOBAL']
X = df[variables_principales]
#X=df.drop(columns=['PUNT_GLOBAL'],axis=1)

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=2)
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(n_estimators=2200, objective="multi:softprob", tree_method='hist', eta=0.1, max_depth=3, enable_categorical=True)
xgb_classifier.fit(Xtrain, Ytrain)

from sklearn.metrics import accuracy_score
preds = xgb_classifier.predict(Xtest)
#print(accuracy_score(Ytest, preds))
import pickle
#pickle.dump(xgb_classifier,open('icfes_clasi.pkl','wb'))


# se balancea el set de datos para NSE dado que NSE1 habian pocos datos, NSE3 muchos
# no es valido dado que no estan balanceados los datos opc 1con los rangos (0-279), (280,-359), (360, inf) genera un accuracy de 0.7343
# opc 2, el accuracy sube a 0.62 con cuartiles (0-223) , (224-261), (262-303), (304 inf)
    


