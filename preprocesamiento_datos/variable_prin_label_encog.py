import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('df_output/balanceado_sinfaltantes.csv')
x= df.drop(['PUNT_GLOBAL','GLOBAL_CATEGORICO'],axis=1)
y = df['GLOBAL_CATEGORICO']

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

#feature_importances_df.to_csv('pca_label_enco.csv',index=False)