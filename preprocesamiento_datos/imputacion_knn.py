"""método de K-Nearest Neighbors (KNN) con el imputer de KNNImputer de la librería scikit-learn. 
Este método imputa los valores faltantes en función de los valores más cercanos (vecinos más cercanos) en el 
espacio multidimensional de las demás observaciones.
OrdinalEncoder: Convierte las categorías en valores numéricos, ya que KNNImputer necesita trabajar con números. Cada categoría se representa como un número.
KNNImputer: Imputa valores faltantes basado en los vecinos más cercanos. El parámetro n_neighbors especifica cuántos vecinos se considerarán para la imputación.
Inverse Transform: Después de la imputación, los valores numéricos se convierten de nuevo a sus categorías originales con 
inverse_transform del OrdinalEncoder."""
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder

# Ejemplo de DataFrame con valores faltantes en columnas categóricas
df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')
# Se elimina la columna ESTU_ETNIA dado que más del 98% de registros son nulos
df_filtrado = df.drop(columns='ESTU_ETNIA')

#Se eliminan las columnas donde se encuentra los resultados por área evaluada, ya que se quiere predecir la variable puntaje global.
otras_puntuaciones =['PUNT_LECTURA_CRITICA','PERCENTIL_LECTURA_CRITICA','DESEMP_LECTURA_CRITICA','PUNT_MATEMATICAS','PERCENTIL_MATEMATICAS','DESEMP_MATEMATICAS','PUNT_C_NATURALES','PERCENTIL_C_NATURALES','DESEMP_C_NATURALES','PUNT_SOCIALES_CIUDADANAS','PERCENTIL_SOCIALES_CIUDADANAS','DESEMP_SOCIALES_CIUDADANAS','PUNT_INGLES','PERCENTIL_INGLES','DESEMP_INGLES','PERCENTIL_GLOBAL']
df_sin_notas = df_filtrado.drop(columns=otras_puntuaciones)
# las columnas ESTU_NSE_INDIVIDUAL y ESTU_HORASSEMANATRABAJA tienen valores numericos, se reemplazan para usar la transformacion.
df_sin_notas['ESTU_NSE_INDIVIDUAL'].replace([1,2,3,4],['NSE1','NSE2','NSE3','NSE4'],inplace = True)
df_sin_notas['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)


# Como KNNImputer funciona con números, primero convertimos las categorías en números usando OrdinalEncoder
encoder = OrdinalEncoder()
df_encoded = pd.DataFrame(encoder.fit_transform(df_sin_notas), columns=df_sin_notas.columns)

# Creación del KNNImputer
knn_imputer = KNNImputer(n_neighbors=3)

# Imputar los valores faltantes
df_imputed_knn = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=df.columns)


# Mostrar el DataFrame imputado
