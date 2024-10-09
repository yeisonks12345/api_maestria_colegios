
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, recall_score

df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')


X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 
y =df['GLOBAL_CATEGORICO']


X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=2)

nb = GaussianNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print('accuracy: ',accuracy_score(y_test, predictions))
print('f1_score: ',f1_score(y_test, predictions, average='weighted'))  # Para manejo de multiclase
print('recall: ',recall_score(y_test, predictions, average='weighted'))

#accuracy NB 0.503485, f1score 0.4087, recall 0.503485