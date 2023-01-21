import streamlit as st
from PIL import Image
import neuron as ne

st.set_page_config(layout="wide")

image = Image.open('neurona.jpg')
st.image(image)

st.header("Simulador de neurona")

# Valores totales de las entradas y los pesos

values = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

# PESOS

st.subheader("Pesos")
weights = []
col_weight = st.columns(values)

for i in range(values):
    weights.append(i)

    with col_weight[i]:
        weights[i] = st.number_input(f"w{i}")

# ENTRADAS 

st.subheader("Entradas")
x = []
col_x = st.columns(values)

for i in range(values):
    x.append(i)

    with col_x[i]:
        x[i] = st.number_input(f"x{i}")

# SESGO Y FUNCIÓN DE ACTIVACIÓN

functions = {'sigmoid': 'sigmoid', 'relu': 'relu', 'tanh': 'tanh'}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Sesgo")
    b = st.number_input("Introduzca el valor del sesgo")
with col2:
    st.subheader("Función de activación")
    func = st.selectbox('Introduzca el valor del sesgo', ['sigmoid', 'relu', 'tanh'])

if st.button("Calcular la salida"):
    my_neuron = ne.Neuron(weights=weights, bias=b, func=functions[func])
    st.text(f"La salida de la neurona es {my_neuron.run(input_data=x)}")


