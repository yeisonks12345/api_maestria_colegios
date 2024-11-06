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
         menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
         #productosAnt= dfMesAnterior['Cantidad'].sum()
         #variacion=productosAnt-productosAct
         st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %',delta="Estudiantes")
       
       with c2:
         entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
         #productosAnt= dfMesAnterior['Cantidad'].sum()
         #variacion=productosAnt-productosAct
         st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta="Estudiantes")
       with c3:
         mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
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
       df_concatenado= pd.concat([input_dfd_copy,input_dfd[['clasificacion']]],axis=1)
       c1,c2 = st.columns([55,45])
       with c1:
          
          edu_madre = df_concatenado['FAMI_EDUCACIONMADRE'].value_counts().reset_index().iloc[0:10]
          fig = px.bar(edu_madre,x='FAMI_EDUCACIONMADRE',y= 'count',title='Distribución educación madre',color='FAMI_EDUCACIONMADRE',labels={'FAMI_EDUCACIONMADRE': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          
          st.plotly_chart(fig,use_container_width=True)
       with c2:

          num_libros = df_concatenado['FAMI_NUMLIBROS'].value_counts().reset_index()
          fig = px.bar(num_libros,x='FAMI_NUMLIBROS',y= 'count',title='Cantidad de libros por familia',color='FAMI_NUMLIBROS',labels={'FAMI_NUMLIBROS': '', 'count': ''},text='count')
          st.plotly_chart(fig,use_container_width=True)
       c3,c4 = st.columns([55,45])
       with c3:
          
          edu_padre = df_concatenado['FAMI_EDUCACIONPADRE'].value_counts().reset_index()[0:10]
          fig = px.bar(edu_padre,x='FAMI_EDUCACIONPADRE',y= 'count',title='Distribución educación padre',color='FAMI_EDUCACIONPADRE',labels={'FAMI_EDUCACIONPADRE': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          
          st.plotly_chart(fig,use_container_width=True)
       with c4:
          
          horas_trabajo = df_concatenado['ESTU_HORASSEMANATRABAJA'].value_counts().reset_index()
          fig = px.bar(horas_trabajo,x='ESTU_HORASSEMANATRABAJA',y= 'count',title='Horas de trabajo estudiante a la semana',color='ESTU_HORASSEMANATRABAJA',labels={'ESTU_HORASSEMANATRABAJA': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          
          st.plotly_chart(fig,use_container_width=True)

       c5,c6 = st.columns(2)
       with c5:
          come_carnepeshu = df_concatenado['FAMI_COMECARNEPESCADOHUEVO'].value_counts().reset_index()
          fig = px.bar(come_carnepeshu,x='FAMI_COMECARNEPESCADOHUEVO',y= 'count',title='Familia come carne, pescado, huevo',color='FAMI_COMECARNEPESCADOHUEVO',labels={'FAMI_COMECARNEPESCADOHUEVO': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          st.plotly_chart(fig,use_container_width=True)  

       with c6:
          
          lectu_diaria = df_concatenado['ESTU_DEDICACIONLECTURADIARIA'].value_counts().reset_index()
          fig = px.bar(lectu_diaria,x='ESTU_DEDICACIONLECTURADIARIA',y= 'count',title='Dedicación lectura diaria',color='ESTU_DEDICACIONLECTURADIARIA',labels={'ESTU_DEDICACIONLECTURADIARIA': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          
          st.plotly_chart(fig,use_container_width=True)
      
       c7,c8,c9 = st.columns(3)
       with c7:
          
          fami_compu = df_concatenado['FAMI_TIENECOMPUTADOR'].value_counts().reset_index()
          fig = px.pie(fami_compu,values= 'count',names='FAMI_TIENECOMPUTADOR',title='Familia tiene computador',color='FAMI_TIENECOMPUTADOR')
          st.plotly_chart(fig,use_container_width=True)
          
          
       with c8:
          fami_internet = df_concatenado['FAMI_TIENEINTERNET'].value_counts().reset_index()
          fig = px.pie(fami_internet,values= 'count',names='FAMI_TIENEINTERNET',title='Familia tiene internet',color='FAMI_TIENEINTERNET')
          st.plotly_chart(fig,use_container_width=True)
       with c9:
          come_leche = df_concatenado['FAMI_COMELECHEDERIVADOS'].value_counts().reset_index()
          fig = px.bar(come_leche,x='FAMI_COMELECHEDERIVADOS',y= 'count',title='Familia come leche y derivados',color='FAMI_COMELECHEDERIVADOS',labels={'FAMI_COMELECHEDERIVADOS': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          st.plotly_chart(fig,use_container_width=True)  


    else:
       st.warning("Por favor, cargue un archivo para continuar.")

def prescripcion():

   st.sidebar.header("Cargue un archivo de Excel con las variables requeridas.")
   st.subheader('Estrategias para mejorar los resultados de las pruebas saber 11.')

   uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])
   options = ["Aumentar cantidad de libros de la familia a mínimo 10", "Madres finalizan bachillerato", "Padres finalizan bachillerato", "Reducir horas de trabajo a la semana", "Familia adquiere computador", "Aumentar dedicación lectura diaria", "Acceder a internet", "Aumentar consumo de Carne, pescado, huevos", "Aumentar consumo Leche y derivados"]
   selected_option = st.selectbox("Elija una estrategia de la lista:", options)


   if uploaded_file is not None:
       input_dfd = pd.read_excel(uploaded_file)
       input_dfd_copy = input_dfd.copy()
       input_dfd_copy_dos = input_dfd.copy()
       load_clf =pickle.load(open('icfes_clasi.pkl','rb'))
       with open('label_encoders.pkl', 'rb') as file:
          loaded_label_encoders = pickle.load(file)
       for col in input_dfd.columns:
             if col in loaded_label_encoders:
               input_dfd[col] = loaded_label_encoders[col].transform(input_dfd[col].astype(str))
       df_original =input_dfd.copy()
       prediction_dos= load_clf.predict(df_original)
       df_original['clasificacion_2'] = prediction_dos
       if selected_option =="Aumentar cantidad de libros de la familia a mínimo 10":
          
          input_dfd['FAMI_NUMLIBROS'].replace({0:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')

       if selected_option =="Madres finalizan bachillerato":
          
          input_dfd['FAMI_EDUCACIONMADRE'].replace({2:8,6:8,7:8,9:8},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')

       if selected_option =="Padres finalizan bachillerato":
          
          input_dfd['FAMI_EDUCACIONPADRE'].replace({2:8,6:8,7:8,9:8},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')  

       if selected_option =="Reducir horas de trabajo a la semana":
          
          input_dfd['ESTU_HORASSEMANATRABAJA'].replace({2:1,4:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')   
       
       if selected_option =="Familia adquiere computador":
          
          input_dfd['FAMI_TIENECOMPUTADOR'].replace({0:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')

       if selected_option =="Aumentar dedicación lectura diaria":
          
          input_dfd['ESTU_DEDICACIONLECTURADIARIA'].replace({0:1,2:1,4:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')

       if selected_option =="Acceder a internet":
          
          input_dfd['FAMI_TIENEINTERNET'].replace({0:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')

       if selected_option =="Aumentar consumo de Carne, pescado, huevos":
          
          input_dfd['FAMI_COMECARNEPESCADOHUEVO'].replace({2:1,0:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')   

       if selected_option =="Aumentar consumo Leche y derivados":
          
          input_dfd['FAMI_COMELECHEDERIVADOS'].replace({2:1,0:1},inplace =True)
          prediction = load_clf.predict(input_dfd)
          input_dfd['clasificacion'] = prediction
          
          c1,c2,c3 = st.columns(3)
          with c1:
             menor_280= round(input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==0].count().iloc[0,]-df_original[df_original['clasificacion_2']==0].count().iloc[0,] 
             st.metric(label="Menor a 280 puntos",value=f'{menor_280:,.0f} %', delta=f'{diferencia:,.0f}')
       
          with c2:
             entre280_360= round(input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==1].count().iloc[0,]-df_original[df_original['clasificacion_2']==1].count().iloc[0,]
         
             st.metric(label="Entre 280 y 360 puntos",value=f'{entre280_360:,.0f} %',delta=f'{diferencia:,.0f}')
          with c3:
             mayor_360= round(input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]/input_dfd['clasificacion'].count()*100,1)
             diferencia = input_dfd[input_dfd['clasificacion']==2].count().iloc[0,]-df_original[df_original['clasificacion_2']==2].count().iloc[0,]

             st.metric(label="Mayor a 360 puntos",value=f'{mayor_360:,.0f} %',delta=f'{diferencia:,.0f}')


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
   prescripcion()