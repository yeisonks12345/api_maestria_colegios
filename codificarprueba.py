import pandas as pd

# Cargar ambas hojas del archivo Excel

codificaciones_df = pd.read_csv('df_output\codificacion_20variables.csv')
dataframe_df = pd.read_excel('comfandi.xlsx')


# Hacemos un merge de cada columna de `dataframe_df` con su codificación en `codificaciones_df`
# Convertir el DataFrame a un formato largo para que cada columna tenga la categoría y el encoding correspondiente
dataframe_long = dataframe_df.melt(var_name='Column', value_name='Category')

# Unir con las codificaciones en base a 'columna' y 'categoria'
dataframe_codificado_long = dataframe_long.merge(codificaciones_df, on=['Column', 'Category'], how='left')

# Volver a dar forma al DataFrame para que esté en formato ancho y solo conserve las codificaciones
dataframe_codificado = dataframe_codificado_long.pivot(index=None, columns='Column', values='Encoding')

# Renombrar las columnas eliminando la multiíndice que genera pivot automáticamente
dataframe_codificado.columns.name = None

# Mostrar el DataFrame codificado
print(dataframe_codificado.head())
