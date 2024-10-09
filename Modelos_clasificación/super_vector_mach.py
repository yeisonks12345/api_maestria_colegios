from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')

X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 
y =df['GLOBAL_CATEGORICO']

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=2)
# Entrenar el modelo
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(Xtrain, Ytrain)

# Predicciones
y_pred = model_svm.predict(Xtest)

# Evaluación
print(confusion_matrix(Ytest, y_pred))
print(classification_report(Ytest, y_pred))
print(accuracy_score(Ytest, y_pred))

#accuracy de: No corre, se prueba por mas de dos horas pero no arroja resultado