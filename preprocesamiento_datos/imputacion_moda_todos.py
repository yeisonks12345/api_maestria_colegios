
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')
# las columnas ESTU_NSE_INDIVIDUAL y ESTU_HORASSEMANATRABAJA tienen valores numericos, se reemplazan para usar la transformacion.

df['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)

#se selecciona las columnas que se usaran para usar el metodo de imputación
columnas_moda = df.drop(columns=['PERIODO','COLE_COD_DANE_ESTABLECIMIENTO','COLE_COD_DANE_SEDE','PUNT_GLOBAL','edad'])
# Creación del SimpleImputer para categorías, estrategia de la moda
imputer = SimpleImputer(strategy='most_frequent')

# Imputar los valores faltantes
df_imputed = pd.DataFrame(imputer.fit_transform(columnas_moda), columns=columnas_moda.columns)
df_final = pd.concat([df[['PERIODO','COLE_COD_DANE_ESTABLECIMIENTO','COLE_COD_DANE_SEDE','PUNT_GLOBAL','edad']],df_imputed],axis=1)
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
#se crea la columna golab categorico a partir de columna puntaje global para categorizar los tres puntajes



df_final['GLOBAL_CATEGORICO'] = pd.cut(df_final['PUNT_GLOBAL'], 
                         bins=[0, 280,360, 500],  # Definir los cortes de los rangos
                         labels=[0, 1, 2],  # Etiquetas: 0, 1, 2
                         include_lowest=True)  # Incluir el límite inferior en el primer rango
     
#no se usan métodos para balancear.
 

#Se envía el dataframe a la carpeta df_output
df_final.to_csv('df_output/imputacion_todos_moda/imputacion_todos_moda.csv',index=False)

