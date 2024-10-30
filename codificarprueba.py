import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

new_df = pd.read_excel('comfandi2.xlsx')
with open('label_encoders.pkl', 'rb') as file:
    loaded_label_encoders = pickle.load(file)

for col in new_df.columns:
    if col in loaded_label_encoders:
        new_df[col] = loaded_label_encoders[col].transform(new_df[col].astype(str))

print("\nNuevo DataFrame transformado:")

print(new_df.head())

