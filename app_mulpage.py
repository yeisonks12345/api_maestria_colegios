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
          minimo_tecnico= edu_madre[edu_madre['FAMI_EDUCACIONMADRE'].isin(['Educacion profesional completa','Educacion profesional incompleta','Postgrado','Tecnica o tecnologica completa'])]['count']
          porcentaje_todos = minimo_tecnico.sum()/edu_madre['count'].sum()
          if porcentaje_todos < 0.9:
            decision_umbral ='Menos del 90% de las madres <br>  cuentan con mínimo un técnico'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(edu_madre,x='FAMI_EDUCACIONMADRE',y= 'count',title='Distribución educación madre',color='FAMI_EDUCACIONMADRE',labels={'FAMI_EDUCACIONMADRE': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=3.4,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)
       with c2:

          num_libros = df_concatenado['FAMI_NUMLIBROS'].value_counts().reset_index()
          aumentar_libros= num_libros[num_libros['FAMI_NUMLIBROS'].isin(['26 A 100 LIBROS','MAS DE 100 LIBROS'])]['count']
          porcentaje_todos = aumentar_libros.sum()/num_libros['count'].sum()
          if porcentaje_todos < 0.8:
            decision_umbral = 'Menos del 80% de <br>las familias <br> cuentan con más de <br> 25 libros en casa'
          else: 
            decision_umbral = 'Ninguna'
          
          fig = px.bar(num_libros,x='FAMI_NUMLIBROS',y= 'count',title='Cantidad de libros por familia',color='FAMI_NUMLIBROS',labels={'FAMI_NUMLIBROS': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=2.4,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)
       c3,c4 = st.columns([55,45])
       with c3:
          
          edu_padre = df_concatenado['FAMI_EDUCACIONPADRE'].value_counts().reset_index()[0:10]
          minimo_tecnico= edu_padre[edu_padre['FAMI_EDUCACIONPADRE'].isin(['Educacion profesional completa','Educacion profesional incompleta','Postgrado','Tecnica o tecnologica completa'])]['count']
          porcentaje_todos = minimo_tecnico.sum()/edu_padre['count'].sum()
          if porcentaje_todos < 0.88:
            decision_umbral ='Fomentar acceso <br> educación superior'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(edu_padre,x='FAMI_EDUCACIONPADRE',y= 'count',title='Distribución educación padre',color='FAMI_EDUCACIONPADRE',labels={'FAMI_EDUCACIONPADRE': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=2.8,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)
       with c4:
          
          horas_trabajo = df_concatenado['ESTU_HORASSEMANATRABAJA'].value_counts().reset_index()
          maximo_20horas= horas_trabajo[horas_trabajo['ESTU_HORASSEMANATRABAJA'].isin(['No_trabaja'])]['count']
          porcentaje_todos = maximo_20horas.sum()/horas_trabajo['count'].sum()
          if porcentaje_todos < 0.94:
            decision_umbral ='Reducir horas <br> de trabajo max. 20'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(horas_trabajo,x='ESTU_HORASSEMANATRABAJA',y= 'count',title='Horas de trabajo estudiante a la semana',color='ESTU_HORASSEMANATRABAJA',labels={'ESTU_HORASSEMANATRABAJA': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=2.3,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)

       c5,c6 = st.columns(2)
       with c5:
          come_carnepeshu = df_concatenado['FAMI_COMECARNEPESCADOHUEVO'].value_counts().reset_index()
          minimo_3veces= come_carnepeshu[come_carnepeshu['FAMI_COMECARNEPESCADOHUEVO'].isin(['Todos o casi todos los dias','3 a 5 veces por semana'])]['count']
          porcentaje_todos = minimo_3veces.sum()/come_carnepeshu['count'].sum()
          if porcentaje_todos < 0.97:
            decision_umbral ='Consumo carne, pescado<br> Huevo 3 veces por semana'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(come_carnepeshu,x='FAMI_COMECARNEPESCADOHUEVO',y= 'count',title='Familia come carne, pescado, huevo',color='FAMI_COMECARNEPESCADOHUEVO',labels={'FAMI_COMECARNEPESCADOHUEVO': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=3.5,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)  

       with c6:
          
          lectu_diaria = df_concatenado['ESTU_DEDICACIONLECTURADIARIA'].value_counts().reset_index()
          suma_mas_unahora= lectu_diaria[lectu_diaria['ESTU_DEDICACIONLECTURADIARIA'].isin(['Entre 30 y 60 minutos','Entre 1 y 2 horas','Más de 2 horas'])]['count']
          porcentaje_todos = suma_mas_unahora.sum()/lectu_diaria['count'].sum()
          if porcentaje_todos <= 0.6:
            decision_umbral ='Aumentar dedicación <br> lectura diaria'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(lectu_diaria,x='ESTU_DEDICACIONLECTURADIARIA',y= 'count',title='Dedicación lectura diaria',color='ESTU_DEDICACIONLECTURADIARIA',labels={'ESTU_DEDICACIONLECTURADIARIA': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=2.8,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          ) 
          st.plotly_chart(fig,use_container_width=True)
          
       c7,c8,c9 = st.columns(3)
       with c7:
          
          fami_compu = df_concatenado['FAMI_TIENECOMPUTADOR'].value_counts().reset_index()
          computador= fami_compu[fami_compu['FAMI_TIENECOMPUTADOR'].isin(['SI'])]['count']
          porcentaje_todos = computador.sum()/fami_compu['count'].sum()
          if porcentaje_todos <= 0.99:
            decision_umbral ='Más del 1% no  <br>tiene computador'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.pie(fami_compu,values= 'count',names='FAMI_TIENECOMPUTADOR',title='Familia tiene computador',color='FAMI_TIENECOMPUTADOR')
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=0.9,                                    # Índice o posición en el eje x
            y=-0.1,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            yref="paper",
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)
          
          
       with c8:
          fami_internet = df_concatenado['FAMI_TIENEINTERNET'].value_counts().reset_index()
          internet= fami_internet[fami_internet['FAMI_TIENEINTERNET'].isin(['SI'])]['count']
          porcentaje_todos = internet.sum()/fami_internet['count'].sum()
          if porcentaje_todos <= 0.99:
            decision_umbral ='Acceso a <br> internet en casa.'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.pie(fami_internet,values= 'count',names='FAMI_TIENEINTERNET',title='Familia tiene internet',color='FAMI_TIENEINTERNET')
          fig.add_annotation(
            text="Mejora detectada: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=0.9,                                    # Índice o posición en el eje x
            y=-0.1,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            yref="paper",
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)
       
       with c9:
          come_leche = df_concatenado['FAMI_COMELECHEDERIVADOS'].value_counts().reset_index()
          minimo_3veces= come_leche[come_leche['FAMI_COMELECHEDERIVADOS'].isin(['Todos o casi todos los dias','3 a 5 veces por semana'])]['count']
          porcentaje_todos = minimo_3veces.sum()/come_leche['count'].sum()
          if porcentaje_todos < 0.90:
            decision_umbral ='Consumo <br>leche<br> derivados 3 <br>veces por <br>semana'
          else: 
            decision_umbral = 'Ninguna'
          fig = px.bar(come_leche,x='FAMI_COMELECHEDERIVADOS',y= 'count',title='Familia come leche y derivados',color='FAMI_COMELECHEDERIVADOS',labels={'FAMI_COMELECHEDERIVADOS': '', 'count': ''},text='count')
          fig.update_xaxes(showticklabels=False)
          fig.add_annotation(
            text="Mejora detec.: <br>"+ decision_umbral,  # Mensaje que aparecerá
            x=2.8,                                    # Índice o posición en el eje x
            y=50,                                   # Coordenada en el eje y
            showarrow=False,                         # Mostrar una flecha
            font=dict(size=14, color="blue"),        # Estilo de la fuente
            xref="paper",                   # Referencia relativa al espacio total del gráfico (0 a 1)
            xanchor="right",                # Anclar el texto al borde derecho
            align='right',
            bordercolor="black",  # Color del borde
            borderwidth=2         # Ancho del borde
          )
          st.plotly_chart(fig,use_container_width=True)  


    else:
       st.warning("Por favor, cargue un archivo para continuar.")

def prescripcion():

   st.sidebar.header("Cargue un archivo de Excel con las variables requeridas.")
   st.subheader('Estrategias para mejorar los resultados de las pruebas saber 11.')

   uploaded_file = st.sidebar.file_uploader('cargue su archivo de Excel',type=['xlsx'])
   options = ["Aumentar cantidad de libros en la familia a mínimo 25", "Madres: Culminar mínimo un técnico", "Padres: Culminar mínimo un técnico", "Reducir horas de trabajo a la semana máximo a 20", "Familia adquiere computador", "Aumentar dedicación lectura diaria mínimo una hora", "Acceder a internet", "Aumentar frecuencia de consumo de Carne, pescado, huevo min 3 veces por semana", "Aumentar frecuencia de consumo de leche y derivados min 3 veces por semana"]
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
       if selected_option =="Aumentar cantidad de libros en la familia a mínimo 25":
          
          input_dfd['FAMI_NUMLIBROS'].replace({0:2,1:2},inplace =True)
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

       if selected_option =="Madres: Culminar mínimo un técnico":
          
          input_dfd['FAMI_EDUCACIONMADRE'].replace({2:10,3:10,4:10,6:10,7:10,8:10,9:10,11:10},inplace =True)
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

       if selected_option =="Padres: Culminar mínimo un técnico":
          
          input_dfd['FAMI_EDUCACIONPADRE'].replace({2:10,3:10,4:10,6:10,7:10,8:10,9:10,11:10},inplace =True)
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

       if selected_option =="Reducir horas de trabajo a la semana máximo a 20":
          
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

       if selected_option =="Aumentar dedicación lectura diaria mínimo una hora":
          
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

       if selected_option =="Aumentar frecuencia de consumo de Carne, pescado, huevo min 3 veces por semana":
          
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

       if selected_option =="Aumentar frecuencia de consumo de leche y derivados min 3 veces por semana":
          
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