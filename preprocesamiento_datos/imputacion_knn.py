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
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Ejemplo de DataFrame con valores faltantes en columnas categóricas
df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')
# Se elimina la columna ESTU_ETNIA dado que más del 98% de registros son nulos

df['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)

#se selecciona las columnas que se usaran para usar el metodo de imputación
columnas_moda = df.drop(columns=['PERIODO','COLE_COD_DANE_ESTABLECIMIENTO','COLE_COD_DANE_SEDE','PUNT_GLOBAL','edad'])

# Como KNNImputer funciona con números, primero convertimos las categorías en números usando OrdinalEncoder
encoder = OrdinalEncoder()
df_encoded = pd.DataFrame(encoder.fit_transform(columnas_moda), columns=columnas_moda.columns)

# Creación del KNNImputer
knn_imputer = KNNImputer(n_neighbors=3)

# Imputar los valores faltantes
df_imputed_knn = pd.DataFrame(knn_imputer.fit_transform(df_encoded), columns=columnas_moda.columns)



df_final = pd.concat([df[['PERIODO','COLE_COD_DANE_ESTABLECIMIENTO','COLE_COD_DANE_SEDE','PUNT_GLOBAL','edad']],df_imputed_knn],axis=1)
# Mostrar el DataFrame imputado

#codificar en variables númericas, se convierten variables categoricas en númericas
label_encoder = LabelEncoder()
for col in df_final.columns:
    if df_final[col].dtype == 'object':
        label_encoder.fit(df_final[col].astype(str))
        df_final[col] = label_encoder.fit_transform(df_final[col])

# se transforma la columna puntaje  global y se asigna la clasificación propuesta. 0,1 y 2
"""

#df_imputed.to_csv('df_output/imputacion_moda/imputacion_modav3.csv',index=False)
print(df_final.info())
"""
#Balanceo del set de datos a partir de columna puntaje global



df_final['GLOBAL_CATEGORICO'] = pd.cut(df_final['PUNT_GLOBAL'], 
                         bins=[0, 280,360, 500],  # Definir los cortes de los rangos
                         labels=[0, 1, 2],  # Etiquetas: 0, 1, 2
                         include_lowest=True)  # Incluir el límite inferior en el primer rango
     
#se usan métodos para balancear.
 

#Sub muestreo
df_class0 = df_final[df_final['GLOBAL_CATEGORICO'] == 0]
df_class1 = df_final[df_final['GLOBAL_CATEGORICO'] == 1]
df_class2 = df_final[df_final['GLOBAL_CATEGORICO'] == 2]
# Determinar el tamaño mínimo para balancear todas las clases
min_size = min(len(df_class0), len(df_class1), len(df_class2))
# Submuestrear cada clase a la clase con menos registros
df_class0_downsampled = resample(df_class0, replace=False, n_samples=min_size, random_state=42)
df_class1_downsampled = resample(df_class1, replace=False, n_samples=min_size, random_state=42)
df_class2_downsampled = resample(df_class2, replace=False, n_samples=min_size, random_state=42)

df_balanced = pd.concat([df_class0_downsampled, df_class1_downsampled, df_class2_downsampled])
#Se envía el dataframe a la carpeta df_output
df_balanced.to_csv('df_output/imputa_knn/imputa_knn.csv',index=False)
