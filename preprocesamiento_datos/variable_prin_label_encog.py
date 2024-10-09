import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('df_input\cali_publicos_privados_2017_2022vf.csv')
# las columnas ESTU_NSE_INDIVIDUAL y ESTU_HORASSEMANATRABAJA tienen valores numericos, se reemplazan para usar la transformacion.

df['ESTU_HORASSEMANATRABAJA'].replace([0],['No_trabaja'],inplace = True)

label_encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        label_encoder.fit(df[col].astype(str))
        df[col] = label_encoder.fit_transform(df[col])

x= df.drop(['PUNT_GLOBAL'],axis=1)
y = df['PUNT_GLOBAL']


# Entrenar el modelo RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)
# Obtener la importancia de las características
importances = model.feature_importances_

# Crear un DataFrame con los nombres de las características y sus importancias
feature_importances_df = pd.DataFrame({
    'Feature': x.columns,     # Nombres de las características
    'Importance': importances # Importancias obtenidas
})

# Ordenar el DataFrame por la importancia de las características (opcional)
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

feature_importances_df.to_csv('df_output/feature_impor/pca_label_enco.csv',index=False)