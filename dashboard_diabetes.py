# imports
import streamlit as st 
import numpy as np
import pickle 
from streamlit_option_menu import option_menu

# page config
st.set_page_config(page_title='Assistente de Saude',
                   page_icon='👨‍⚕️',
                   layout='wide')

# carrgamento do modelo treinado
modelo_treinado = pickle.load(open('modelo_treinado.pkl', 'rb'))

# barra de navegação
with st.sidebar:
    selected = option_menu('Diabetes Disease System',
                           ['Diabetes Prediction', 'About'],
                           menu_icon='hospital-fill',
                           icons=['heart', 'activity'],
                           default_index=0)

# diabetes prediction page
if selected == 'Diabetes Prediction':
    c1, c2, c3 = st.columns(3, gap='large')

    with c1:
        #Pregnancies = st.text_input('Number of pregnancies')
        Pregnancies = st.slider('Number of pregnancies', min_value=0.0,
                                 max_value=17.0, step=1.0)
        #SkinThickness = st.text_input('Skin Thinckness value')
        SkinThickness = st.slider('Skin Thinckness value', min_value=0.0, max_value=99.0, step=0.5)
        #DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function value')
        DiabetesPedigreeFunction = st.slider('Diabetes pedigree function value', min_value=0.078, max_value=2.420, step=0.04)

    with c2:
        #Glucose = st.text_input('Glucose level')
        Glucose = st.slider('Glucose level', min_value=0.0, max_value=199.0, step=0.5)
        #Insulin = st.text_input('Insulin level')
        Insulin = st.slider('Insulin level', min_value=0.0, max_value=846.0, step=0.5)
        #Age = st.text_input('Age of the person')
        Age = st.slider('Age of the person', min_value=21.0, max_value=81.0, step=1.0)
    
    with c3:
        #BloodPressure = st.text_input('Blood Pressure value')
        BloodPressure = st.slider('Blood Pressure value', min_value=0.0, max_value=122.0, step=1.0)
        #BMI = st.text_input('BMI value')
        BMI = st.slider('BMI value', min_value=0.0, max_value=67.1, step=0.1)

    # prediction
    #diabetes_diagnosis = ''

    if st.button('Diabetes test result'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]
        
        user_input = np.array(user_input).reshape(1, -1)

        diab_pred = modelo_treinado.predict(user_input)
        diab_pred_prob = modelo_treinado.predict_proba(user_input)

        if diab_pred[0] == 0:
            st.write(f'Resultado: :blue[{diab_pred[0]}] | :green[Existe {diab_pred_prob[0][0]*100:.2f}% de chance de o(a) paciente não ser diabético(a)]')
        else:
            st.write(f'Resultado: :red[{diab_pred[0]}] | :orange[Existe {diab_pred_prob[0][1]*100:.2f}% de chance de o(a) paciente ser diabético(a)]')

elif selected == 'About':
    st.subheader('Informações gerais sobre o App')