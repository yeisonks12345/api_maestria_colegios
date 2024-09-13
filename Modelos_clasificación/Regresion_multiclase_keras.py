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

df = pd.read_csv('df_output\datos_codifcadobalanceados.csv')
variables_principales= ['INSE','EDAD'
                        ,'ESTU_DEDICACIONLECTURADIARIA','ESTU_DEDICACIONINTERNET']
df_y = df[['PUNT_GLOBAL']]
def y_cambio(i):
    if i<=279:
        return 0
    elif i >=280 and i <= 359:
        return 1
 
    
    elif i >=360:
        return 2

df_y['PUNT_GLOBAL'] = df_y['PUNT_GLOBAL'].apply(y_cambio)
Y_num= df_y['PUNT_GLOBAL'].values
Y = np_utils.to_categorical(Y_num,3)
X = df[variables_principales].values



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
