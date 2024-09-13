#en este script se codifican los datos pasan de categoricos a numericos,
#las columna estu_nse individual y ESTU_HORASSEMANATRABAJA tenian numeros como el 0,1,2,3 toca reemplazarlos. 

import pandas as pd
df_transfor = pd.read_excel('df_output\datos_priv_balanceados.xlsx')
df_transfor["INSE"] = df_transfor["INSE"].astype(float, errors="raise")
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
for col in df_transfor.columns:
    if df_transfor[col].dtype == 'object':
        label_encoder.fit(df_transfor[col].astype(str))
        df_transfor[col] = label_encoder.fit_transform(df_transfor[col])
df_transfor.to_csv('df_output\datos_codifcadobalanceados.csv',index=False)
#print(df_transfor.info())

