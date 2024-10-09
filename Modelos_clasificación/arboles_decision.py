import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score

df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')


X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 
y =df['GLOBAL_CATEGORICO']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)

# Predicciones
predictions = model_dt.predict(X_test)

# Evaluación
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print('accuracy: ',accuracy_score(y_test, predictions))
print('f1_score: ',f1_score(y_test, predictions, average='weighted'))  # Para manejo de multiclase
print('recall: ',recall_score(y_test, predictions, average='weighted'))
 
#arboles de decision genera un accuracy de 0.5340, f1score 0.5312, recall 0.5340 