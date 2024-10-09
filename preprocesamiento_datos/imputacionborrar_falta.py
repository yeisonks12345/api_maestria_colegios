import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')


df['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)

#Metodo de imputacion 1, eliminar datos faltantes
df_sin_faltantes = df.dropna()

#codificar en variables númericas, se convierten variables categoricas en númericas
label_encoder = LabelEncoder()
for col in df_sin_faltantes.columns:
    if df_sin_faltantes[col].dtype == 'object':
        label_encoder.fit(df_sin_faltantes[col].astype(str))
        df_sin_faltantes[col] = label_encoder.fit_transform(df_sin_faltantes[col])


#Balanceo del set de datos a partir de columna puntaje global

"""
se balancean las variables con base en la columna puntaje global, 
se crean tres clases para la variable a predecir "puntaje global":
Rango Menor o igual a 280','Rango entre 281 y 360 puntos', 'Rango mayor a 361 puntos
"""
# se transforma la columna puntaje  global y se asigna la clasificación propuesta. 0,1 y 2
df_sin_faltantes['GLOBAL_CATEGORICO'] = pd.cut(df_sin_faltantes['PUNT_GLOBAL'], 
                         bins=[0, 280,360, 500],  # Definir los cortes de los rangos
                         labels=[0, 1, 2],  # Etiquetas: 0, 1, 2
                         include_lowest=True)  # Incluir el límite inferior en el primer rango
     
#se usan métodos para balancear.
 
"""
Submuestreo (Under-sampling)
Se identifica que la variable a predecir punt global esta desbalanceada, por consiguiente se 
usa el metodo de submuestreo para nivelar las clases.
"""
#Sub muestreo
df_class0 = df_sin_faltantes[df_sin_faltantes['GLOBAL_CATEGORICO'] == 0]
df_class1 = df_sin_faltantes[df_sin_faltantes['GLOBAL_CATEGORICO'] == 1]
df_class2 = df_sin_faltantes[df_sin_faltantes['GLOBAL_CATEGORICO'] == 2]
# Determinar el tamaño mínimo para balancear todas las clases
min_size = min(len(df_class0), len(df_class1), len(df_class2))
# Submuestrear cada clase a la clase con menos registros
df_class0_downsampled = resample(df_class0, replace=False, n_samples=min_size, random_state=42)
df_class1_downsampled = resample(df_class1, replace=False, n_samples=min_size, random_state=42)
df_class2_downsampled = resample(df_class2, replace=False, n_samples=min_size, random_state=42)

df_balanced = pd.concat([df_class0_downsampled, df_class1_downsampled, df_class2_downsampled])
#Se envía el dataframe a la carpeta df_output
df_balanced.to_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv',index=False)


