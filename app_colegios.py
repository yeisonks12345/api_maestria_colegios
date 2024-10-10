#el modelo actual genera un accuracy de 0.67
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import plotly.express as px
#configurar pagina streamlit

st.set_page_config(page_title="App colegios predicción",
                   layout="centered",
                   initial_sidebar_state="auto")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f0f5;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True
)
#logo de la universidad
logo = Image.open("images\LogotipoESCUELA.png") 
st.sidebar.image(logo, use_column_width=True)  # En la barra lateral



#definimos titulo

st.markdown("<h1 style='text-align: center; color: blue;'>App para clasificar resultados pruebas Saber</h1>", unsafe_allow_html=True)
#st.markdown("""<div style='text-align: justify;'>A partir de información sociodemográfica de los estudiantes y la familia, la aplicación pronostica y clasifica en tres rangos los posibles resultados de las pruebas Saber.</div>""", unsafe_allow_html=True)

st.markdown("""
<div style='border: 2px solid #000; padding: 20px; border-radius: 10px;'>
    <div style='text-align: justify;'>
        A partir de información socioeconómica de los estudiantes y la familia, que aún no han presentado las pruebas saber once en Colombia, la aplicación pronostica y clasifica en tres rangos los posibles resultados de las pruebas. Se usa el algoritmo de machine learning XGBoost classifier para el despliegue de la aplicación.
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style='margin-top: 10px;'>
    <strong>Rango 1:</strong> Menor a 280 puntos<br>
    <strong>Rango 2:</strong> Entre 281 y 360 puntos<br>
    <strong>Rango 3:</strong> Mayor a 360 
    <hr style='margin: 20px 0; border: 1px solid #ccc;'>        
</div>
""", unsafe_allow_html=True)


st.sidebar.header("Cargue un archivo de Excel con las variables requeridas.")
st.subheader('En la lista desplegable, podrá seleccionar entre tres rangos, se genera una lista que podrá descargar en Excel.')
rangos = st.selectbox('',['Rango Menor a 280','Rango entre 280 y 360 puntos', 'Rango mayor a 360 puntos'])

uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])

if uploaded_file is not None:

    input_dfd = pd.read_excel(uploaded_file)
    load_clf =pickle.load(open('icfes_clasi.pkl','rb'))

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


# se debe definir que variables se usaran en el despliegue y garantizar que el label encoding sea igual para todos, se us las mismas etiquetas