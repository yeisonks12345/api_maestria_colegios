#error de dll al usar keras y tensor se resuelve con pip uninstall tensorflow
#pip install tensorflow==2.12.0 --upgrade
# para esta red neuronal es necesario que la variable y se divida a través de onehotencoding, 
#lo que hace es crear una columna para cada categoria 
"""
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
df = pd.read_csv('df_output/balanceado_sinfaltantes.csv')
X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL','INSE','ESTU_CONSECUTIVO','ESTU_NSE_ESTABLECIMIENTO','ESTU_NSE_ESTABLECIMIENTO','COLE_CODIGO_ICFES','COLE_DEPTO_UBICACION','COLE_COD_DEPTO_UBICACION','COLE_MCPIO_UBICACION','ESTU_ESTUDIANTE','COLE_COD_MCPIO_UBICACION','ESTU_PRIVADO_LIBERTAD','ESTU_COD_MCPIO_PRESENTACION','ESTU_ESTADOINVESTIGACION','ESTU_MCPIO_PRESENTACION','ESTU_DEPTO_PRESENTACION','ESTU_COD_DEPTO_PRESENTACION','ESTU_COD_RESIDE_DEPTO','ESTU_DEPTO_RESIDE'],axis=1) 
Y_num =df['GLOBAL_CATEGORICO'].values

Y = np_utils.to_categorical(Y_num,3)


np.random.seed(1)
input_dim = X.shape[1]
output_dim = Y.shape[1]
modelo = Sequential()
modelo.add(Dense(output_dim,input_dim=input_dim,activation='softmax'))

sgd = SGD(lr=0.1)
modelo.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
n_its = 2000
batch_size = X.shape[0]
historia = modelo.fit(X,Y,epochs=n_its,batch_size=batch_size,verbose=2)
#accuracy 0.33
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from keras.utils import np_utils
# Cargar un dataset de ejemplo

df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')


X =df.drop(['GLOBAL_CATEGORICO','PUNT_GLOBAL'],axis=1) 

Y_num =df['GLOBAL_CATEGORICO'].values

#y = np_utils.to_categorical(Y_num,3)


# Convertir la variable objetivo a formato one-hot encoding
y_categorical = to_categorical(Y_num, num_classes=3)

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# Construcción de un modelo simple con Keras
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 categorías para la regresión multiclase

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Hacer predicciones sobre los datos de prueba
y_pred = model.predict(X_test)

# Convertir las predicciones de probabilidades a etiquetas de clases
y_pred_classes = np.argmax(y_pred, axis=1)

# Convertir las etiquetas verdaderas de one-hot encoding a etiquetas de clases
y_true_classes = np.argmax(y_test, axis=1)

# Calcular F1-score y recall con sklearn
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')

# Mostrar los resultados
print(f"F1-Score (weighted): {f1}")
print(f"Recall (weighted): {recall}")

#accuracy 0.3293, f1score 0.1667, recall 0.3334