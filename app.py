import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pickle import dump
from pickle import load
from pyngrok import ngrok
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, mutual_info_classif
# Judul halaman
st.title('Aplikasi Data Science')
st.subheader("Teknik Pengambilan Sampel Darah (Serum atau Plasma) dengan menggunakan laboratorium test")

st.markdown("- Alat Pengambilan Sampel:")
st.write("Jarum suntik atau sarung tangan pengambil darah (vacutainer).")

st.markdown("- Teknik Pengambilan Sampel:")
st.write("Sampel darah biasanya diambil dari vena di lengan, seringkali dari vena di bagian dalam siku. "
        "Pada beberapa uji, dapat digunakan darah kapiler dari ujung jari.")

st.markdown("- Persiapan:")
st.write("Pasien biasanya diminta untuk berpuasa sebelum pengambilan darah pada uji tertentu.")
st.subheader('Implementasi')

X_new = load(open('x.pkl', 'rb'))
y = load(open('y.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
X_test_scaled = load(open('scaled.pkl', 'rb'))
scaler=load(open('scaler.pkl', 'rb'))
st.write('Masukkan data untuk melakukan prediksi')
input_data=[]
input_sex = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
if input_sex == 'Male':
        input_data.append(1)
else:
        input_data.append(2)
input_data.append(st.number_input('Age'))
input_data.append(st.number_input('ALB'))
input_data.append(st.number_input('ALP'))
input_data.append(st.number_input('ALT'))
input_data.append(st.number_input('AST'))
input_data.append(st.number_input('BIL'))
input_data.append(st.number_input('CHE'))
input_data.append(st.number_input('CHOL'))
input_data.append(st.number_input('CREA'))
input_data.append(st.number_input('GGT'))
input_data.append(st.number_input('PROT'))

if st.button('Prediksi'):
    # Pilihan algoritma
    rf = load(open('RF.pkl', 'rb'))
    input_data_scaled = scaler.transform([input_data])
    predicted_class = rf.predict(input_data_scaled)
    predicted_class_label = "healthy" if predicted_class[0] == 0 else "hepatitis"
    st.subheader('Hasil Prediksi')
    st.write(predicted_class_label)




