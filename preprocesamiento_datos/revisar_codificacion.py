import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
"""
df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')


df['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)
df_colum = df[['COLE_CALENDARIO','COLE_JORNADA','FAMI_NUMLIBROS','COLE_CARACTER','COLE_BILINGUE','FAMI_EDUCACIONMADRE','FAMI_ESTRATOVIVIENDA','edad','ESTU_GENERO','FAMI_TIENEMOTOCICLETA','FAMI_EDUCACIONPADRE','FAMI_TIENEAUTOMOVIL','ESTU_HORASSEMANATRABAJA','FAMI_TIENECOMPUTADOR','FAMI_COMECARNEPESCADOHUEVO','ESTU_DEDICACIONLECTURADIARIA','ESTU_TIPODOCUMENTO','FAMI_SITUACIONECONOMICA','FAMI_COMELECHEDERIVADOS','FAMI_TIENEINTERNET']]
#Metodo de imputacion 1, eliminar datos faltantes
df_sin_faltantes = df_colum.dropna()
"""
df_sin_faltantes=pd.read_excel('miraflores_sinfaltantes.xlsx')
#codificar en variables númericas, se convierten variables categoricas en númericas
label_encoder = LabelEncoder()
encoding_dict = {}
"""
for col in df_sin_faltantes.columns:
    if df_sin_faltantes[col].dtype == 'object':
        label_encoder.fit(df_sin_faltantes[col].astype(str))
        df_sin_faltantes[col] = label_encoder.fit_transform(df_sin_faltantes[col])

"""
for col in df_sin_faltantes.columns:
    if df_sin_faltantes[col].dtype == 'object':
        df_sin_faltantes[col] = label_encoder.fit_transform(df_sin_faltantes[col].astype(str))
        # Guardar el mapeo en el diccionario
        encoding_dict[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Convertir el diccionario en un DataFrame para visualizar las codificaciones
df_encodings = pd.DataFrame([
    {'Column': col, 'Category': category, 'Encoding': encoding}
    for col, mappings in encoding_dict.items()
    for category, encoding in mappings.items()
])

# Mostrar el DataFrame resultante con las codificaciones
df_encodings.to_csv('pruebacodifciacmiraflores.csv',index=False)
