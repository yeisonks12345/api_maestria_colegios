#error de dll al usar keras y tensor se resuelve con pip uninstall tensorflow
#pip install tensorflow==2.12.0 --upgrade
# para esta red neuronal es necesario que la variable y se divida a través de onehotencoding, 
#lo que hace es crear una columna para cada categoria 
# Genera un accuracy de 0.61, se quita la variable de codigo dane
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