#Se requieren 34 varaibles para no depender del INSE y del NSE, genera un score de 0.73
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
#configurar pagina streamlit

st.set_page_config(page_title="App colegios predicción",
                   layout="centered",
                   initial_sidebar_state="auto")

#definimos titulo
st.title("App para clasificar resultados pruebas saber")
st.markdown(""" Esta aplicacion clasifica en tres segmentos los posibles resultados de las prubas saber a partir de datos de entrada como condicion socieconomica,
            horas de dedicación a la lectura, entre otras.""")
st.sidebar.header("Datos suministrados por el usuario, cargue un archivo de Excel con las variables requeridas por el modelo")
st.header('Escoja alguno de los siguientes rangos para generar el listado: ')
rangos = st.selectbox('',['Rango Menor a 279','Rango entre 280 y 359 puntos', 'Rango mayor a 360 puntos'])

uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])
input_dfd = pd.read_excel(uploaded_file)
load_clf =pickle.load(open('icfes_clasi.pkl','rb'))

prediction = load_clf.predict(input_dfd)
input_dfd['clasificacion'] = prediction

#st.write("Con base en los parametros indicados los estudiantes que podrian obtener un resultado igual o menor a 279 puntos son:")
input_dfd['clasificacion'].replace([0],['menor a 279'],inplace=True)
input_dfd['clasificacion'].replace([1],['entre 280 y 359'],inplace=True)
input_dfd['clasificacion'].replace([2],['mayor a 360'],inplace=True)
if rangos == 'Rango Menor a 279':

    st.write(input_dfd[input_dfd['clasificacion']=='menor a 279'].reset_index())
elif rangos == 'Rango entre 280 y 359 puntos':
    st.write(input_dfd[input_dfd['clasificacion']=='entre 280 y 359'].reset_index())
elif rangos == 'Rango mayor a 360 puntos':
    st.write(input_dfd[input_dfd['clasificacion']=='mayor a 360'].reset_index())


#st.write("Con base en los parametros indicados los estudiantes que podrian obtener un resultado entre 280 y 359 puntos son:")



#st.write("Con base en los parametros indicados los estudiantes que podrian obtener un resultado mayor a 360 puntos son:")


#(0-279), (280,-359), (360, inf)
