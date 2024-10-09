from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import f1_score, recall_score

df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')


X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 
y =df['GLOBAL_CATEGORICO']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

ran_forest = RandomForestClassifier(n_estimators=19,random_state=2016,min_samples_leaf=6,)

ran_forest.fit(X_train,y_train)

predictions = ran_forest.predict(X_test)

print('accuracy: ',accuracy_score(y_test, predictions))
print('f1_score: ',f1_score(y_test, predictions, average='weighted'))  # Para manejo de multiclase
print('recall: ',recall_score(y_test, predictions, average='weighted'))

#genera un accuracy de 0.6380, f1score 0.632682, recall 0.638069
