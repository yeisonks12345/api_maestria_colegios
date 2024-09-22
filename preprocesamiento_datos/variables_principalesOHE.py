import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Cargar el conjunto de datos (con variables categóricas)
df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')
# Se elimina la columna ESTU_ETNIA dado que más del 98% de registros son nulos
df_filtrado = df.drop(columns='ESTU_ETNIA')

#Se eliminan las columnas donde se encuentra los resultados por área evaluada, ya que se quiere predecir la variable puntaje global.
otras_puntuaciones =['PUNT_LECTURA_CRITICA','PERCENTIL_LECTURA_CRITICA','DESEMP_LECTURA_CRITICA','PUNT_MATEMATICAS','PERCENTIL_MATEMATICAS','DESEMP_MATEMATICAS','PUNT_C_NATURALES','PERCENTIL_C_NATURALES','DESEMP_C_NATURALES','PUNT_SOCIALES_CIUDADANAS','PERCENTIL_SOCIALES_CIUDADANAS','DESEMP_SOCIALES_CIUDADANAS','PUNT_INGLES','PERCENTIL_INGLES','DESEMP_INGLES','PERCENTIL_GLOBAL']
df_sin_notas = df_filtrado.drop(columns=otras_puntuaciones)
# las columnas ESTU_NSE_INDIVIDUAL y ESTU_HORASSEMANATRABAJA tienen valores numericos, se reemplazan para usar la transformacion.
df_sin_notas['ESTU_NSE_INDIVIDUAL'].replace([1,2,3,4],['NSE1','NSE2','NSE3','NSE4'],inplace = True)
df_sin_notas['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)

#Metodo de imputacion 1, eliminar datos faltantes
df_sin_faltantes = df_sin_notas.dropna()
columnas_eliminar= ['ESTU_CONSECUTIVO','COLE_NOMBRE_ESTABLECIMIENTO','COLE_NOMBRE_SEDE']
df_sin_columnas = df_sin_faltantes.drop(columns=columnas_eliminar)
#Seleccionar solo objectos
df_object = df_sin_columnas.select_dtypes(include=['object'])


# Aplicar One-Hot Encoding para las variables categóricas

df_encoded = pd.get_dummies(df_object, drop_first=True)

# Escalar los datos para que todas las variables tengan la misma magnitud
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded)

# Aplicar PCA
pca = PCA(n_components=None)  # n_components=None para conservar todos los componentes
pca.fit(df_scaled)

# Ver la cantidad de varianza explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_

# Crear un DataFrame para visualizar los resultados
pca_df = pd.DataFrame({
    'Componente Principal': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Varianza Explicada': explained_variance
})

# Imprimir la varianza explicada por cada componente principal
print(pca_df)

# Ver las contribuciones de las variables originales (incluidas las codificadas) a cada componente principal
loadings = pd.DataFrame(pca.components_, columns=df_encoded.columns)
print("\nContribuciones de las variables a cada Componente Principal:")
loadings.to_csv('pca.csv',index=False)
