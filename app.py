import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("pizzas.csv")

model = LinearRegression()
x = df[["diametro"]]
y = df[["preco"]]

model.fit(x, y)

st.title("Predicting the value of a pizza")
st.divider()

diameter = st.number_input("Enter the pizza diameter size")

if diameter:
    expectedPrice = model.predict([[diameter]])[0][0]
    st.write(f"The price of a pizza with a diameter of {diameter:.2f} cm is R${expectedPrice:.2f}")
