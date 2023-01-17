import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

st.dataframe(df)

if st.checkbox("Show line plot"):
    st.line_chart(df)

if st.checkbox("Show bar plot"):
    st.bar_chart(df)
