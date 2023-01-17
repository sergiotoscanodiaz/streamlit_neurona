import streamlit as st
from PIL import Image
import neuron as ne

st.set_page_config(layout="wide")

image = Image.open('neurona.jpg')
st.image(image)

st.header("Simulador de neurona")

num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

st.subheader("Pesos")
w = []
col_w = st.columns(num)

for i in range(num):
    w.append(i)

    with col_w[i]:
        w[i] = st.number_input(f"w_input_{i}", label_visibility="collapsed")

st.subheader("Entradas")
x = []
col_x = st.columns(num)

for i in range(num):
    x.append(i)

    with col_x[i]:
        x[i] = st.number_input(f"x_input_{i}", label_visibility="collapsed")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sesgo")
    b = st.number_input("Introduzca el valor del sesgo")
with col2:
    st.subheader("Función de activación")
    function = st.selectbox('Introduzca el valor del sesgo', ['Sigmoide', 'ReLU', 'Tangente hiperbólica'])


FUNCTIONS = {'Sigmoide': 'sigmoid', 'ReLU': 'relu', 'Tangente hiperbólica': 'tanh'}

if st.button("Calcular la salida"):
    my_neuron = ne.Neuron(weights=w, bias=b, func=FUNCTIONS[function])
    st.text(f"La salida de la neurona es {my_neuron.run(input_data=x)}")

