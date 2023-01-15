import streamlit as st
from scipy.optimize import minimize

def optimize():
    st.title("Optimization Demo")
    algorithm = st.selectbox("Select optimization algorithm", ["BFGS", "L-BFGS-B", "SLSQP"])
    func = st.text_input("Enter a function to optimize")
    x0 = st.number_input("Enter initial value for x")
    res = minimize(lambda x: eval(func), x0, method=algorithm)
    st.write("Optimized value: ", res.x)

if __name__ == "__main__":
    optimize()