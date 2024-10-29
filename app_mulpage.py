import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
def inicio():
    st.title('Aplicación para estimar los resultados de las pruebas saber 11')
    st.write('Bienvenido')
    st.write('Usa el menu de la izquierda para la navegacion')
    logo = Image.open("images\LogotipoESCUELA.png") 
    st.image(logo, width=400)

def prediccion():
    st.sidebar.header("Cargue un archivo de Excel con las variables requeridas.")
    st.subheader('En la lista desplegable, podrá seleccionar entre tres rangos, se genera una lista que podrá descargar en Excel.')
    rangos = st.selectbox('',['Rango Menor a 280','Rango entre 280 y 360 puntos', 'Rango mayor a 360 puntos'])

    uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])

    if uploaded_file is not None:

       input_dfd = pd.read_excel(uploaded_file)
       load_clf =pickle.load(open('icfes_clasi.pkl','rb'))
       label_encoder = LabelEncoder()
       for col in input_dfd.columns:
          if input_dfd[col].dtype == 'object':
             label_encoder.fit(input_dfd[col].astype(str))
             input_dfd[col] = label_encoder.fit_transform(input_dfd[col])


       prediction = load_clf.predict(input_dfd)
       input_dfd['clasificacion'] = prediction

#st.write("Con base en los parametros indicados los estudiantes que podrian obtener un resultado igual o menor a 279 puntos son:")
       input_dfd['clasificacion'].replace([0],['menor a 280'],inplace=True)
       input_dfd['clasificacion'].replace([1],['entre 280 y 360'],inplace=True)
       input_dfd['clasificacion'].replace([2],['mayor a 360'],inplace=True)
       if rangos == 'Rango Menor a 280':

          st.write(input_dfd[input_dfd['clasificacion']=='menor a 280'].reset_index())
       elif rangos == 'Rango entre 280 y 360 puntos':
          st.write(input_dfd[input_dfd['clasificacion']=='entre 280 y 360'].reset_index())
       elif rangos == 'Rango mayor a 360 puntos':
          st.write(input_dfd[input_dfd['clasificacion']=='mayor a 360'].reset_index())
       fig = px.histogram(input_dfd, x='clasificacion', title='Distribución de la Clasificación')
       st.plotly_chart(fig)    
    else:
       st.warning("Por favor, cargue un archivo para continuar.")


st.sidebar.title('Navegacion')
pagina = st.sidebar.radio('Selecciona una página',['Inicio','Predicción','Mejora'])

if pagina=="Inicio":
    inicio()

elif pagina=='Predicción':
   prediccion()