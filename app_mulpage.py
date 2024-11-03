import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    st.subheader('Predicción resultados pruebas saber 11.')
    

    uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])

    if uploaded_file is not None:

       input_dfd = pd.read_excel(uploaded_file)
       input_dfd_copy = input_dfd.copy()
       load_clf =pickle.load(open('icfes_clasi.pkl','rb'))
       
       with open('label_encoders.pkl', 'rb') as file:
          loaded_label_encoders = pickle.load(file)

       for col in input_dfd.columns:
          if col in loaded_label_encoders:
              input_dfd[col] = loaded_label_encoders[col].transform(input_dfd[col].astype(str))


       prediction = load_clf.predict(input_dfd)
       input_dfd['clasificacion'] = prediction
       c1,c2,c3 = st.columns(3)
       with c1:
         menor_280= round(input_dfd[input_dfd['clasificacion']==0].count()[0]/input_dfd['clasificacion'].count()*100,1)
         #productosAnt= dfMesAnterior['Cantidad'].sum()
         #variacion=productosAnt-productosAct
         st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %',delta="Estudiantes")
       
       with c2:
         entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count()[0]/input_dfd['clasificacion'].count()*100,1)
         #productosAnt= dfMesAnterior['Cantidad'].sum()
         #variacion=productosAnt-productosAct
         st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta="Estudiantes")
       with c3:
         mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count()[0]/input_dfd['clasificacion'].count()*100,1)
         #productosAnt= dfMesAnterior['Cantidad'].sum()
         #variacion=productosAnt-productosAct
         st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta="Estudiantes")

# graficos con plotly
       c1,c2 = st.columns([60,40])
       with c1:
          conteo_puntaje = input_dfd['clasificacion'].value_counts().reset_index().sort_values(by='count',ascending=False)
          conteo_puntaje.replace({0:'Menor a 280',1:'Entre 280 y 360',2:'Mayor a 360'},inplace=True)
          fig = px.bar(conteo_puntaje, x='clasificacion', y='count',title='Distribución de la Clasificación',color='clasificacion',text='count')
          st.plotly_chart(fig,use_container_width=True)
       with c2:

          st.write('\n\n')
          st.write('\n\n')
          st.write('\n\n**Listado predicción resultados**')
          df_concatenado= pd.concat([input_dfd_copy,input_dfd[['clasificacion']]],axis=1)
          st.dataframe(df_concatenado, use_container_width=True, hide_index=True)
              
    else:
       st.warning("Por favor, cargue un archivo para continuar.")

def caracterizacion():
    st.sidebar.header("Cargue un archivo de Excel con las variables requeridas.")
    st.subheader('Caracterización del grupo')
    

    uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])

    if uploaded_file is not None:

       input_dfd = pd.read_excel(uploaded_file)
       input_dfd_copy = input_dfd.copy()
       load_clf =pickle.load(open('icfes_clasi.pkl','rb'))
       
       with open('label_encoders.pkl', 'rb') as file:
          loaded_label_encoders = pickle.load(file)

       for col in input_dfd.columns:
          if col in loaded_label_encoders:
              input_dfd[col] = loaded_label_encoders[col].transform(input_dfd[col].astype(str))


       prediction = load_clf.predict(input_dfd)
       input_dfd['clasificacion'] = prediction
       c1,c2,c3 = st.columns(3)
       with c1:
          gender_counts = input_dfd['ESTU_GENERO'].value_counts().reset_index()
          fig = px.pie(gender_counts,values= 'count',names='ESTU_GENERO',title='Distribución por genero',color='ESTU_GENERO')
          st.plotly_chart(fig,use_container_width=True)
       with c2:
          df_concatenado= pd.concat([input_dfd_copy,input_dfd[['clasificacion']]],axis=1)
          num_libros = df_concatenado['FAMI_NUMLIBROS'].value_counts().reset_index()
          fig = px.bar(num_libros,x='FAMI_NUMLIBROS',y= 'count',title='Distribución cantidad de libros',color='FAMI_NUMLIBROS')
          st.plotly_chart(fig,use_container_width=True)
       with c3:
          gender_counts = input_dfd['ESTU_GENERO'].value_counts().reset_index()
          fig = px.pie(gender_counts,values= 'count',names='ESTU_GENERO',title='Distribución por genero',color='ESTU_GENERO')
          st.plotly_chart(fig,use_container_width=True)
    
    else:
       st.warning("Por favor, cargue un archivo para continuar.")

st.sidebar.title('Navegacion')
pagina = st.sidebar.radio('Selecciona una página',['Inicio','Caracterización','Predicción','Prescripción'])

if pagina=="Inicio":
    inicio()

elif pagina=='Predicción':
   prediccion()

elif pagina=='Caracterización':
   caracterizacion()
elif pagina=='Prescripción':
   pass