import pandas as pd
df = pd.read_csv('df_output/borrando_faltantes/balanceado_sinfaltantes.csv')
#se identifica el colegio con mayor puntaje promedio, el cual es el codificado como 79
#print(df.groupby('COLE_NOMBRE_ESTABLECIMIENTO')['PUNT_GLOBAL'].mean().reset_index().sort_values(by='PUNT_GLOBAL', ascending=False))
# se crea un dataframe para el coelgio 79
colegio_79 = df[df['COLE_NOMBRE_ESTABLECIMIENTO']==79]

colegio_79_variables = colegio_79[['FAMI_NUMLIBROS','FAMI_EDUCACIONMADRE','FAMI_EDUCACIONPADRE','ESTU_HORASSEMANATRABAJA','FAMI_TIENECOMPUTADOR','ESTU_DEDICACIONLECTURADIARIA','FAMI_COMECARNEPESCADOHUEVO','FAMI_COMELECHEDERIVADOS','FAMI_TIENEINTERNET']]
print(colegio_79_variables.head())