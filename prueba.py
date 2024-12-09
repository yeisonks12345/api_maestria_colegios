import pickle
import pandas as pd
input_df = pd.read_excel('Datos reales miraflores.xlsx')
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in input_df.columns:
    if input_df[col].dtype == 'object':
        label_encoder.fit(input_df[col].astype(str))
        input_df[col] = label_encoder.fit_transform(input_df[col])
load_clf =pickle.load(open('icfes_clasi.pkl','rb'))

prediction = load_clf.predict(input_df)

input_df['clasi'] = prediction
input_df.to_excel('resultados.xlsx')

print(input_df.head())