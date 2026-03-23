import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Boeing Analytics Tool", layout="wide")

st.title("✈️ Aerospace Performance & Drag Polar Analyzer")
st.markdown("Automated visualization tool for aerodynamic efficiency and performance metrics.")

# Sidebar Controls
st.sidebar.header("Aircraft Design Parameters")
AR = st.sidebar.slider("Aspect Ratio (AR)", 5.0, 20.0, 10.0)
e = st.sidebar.slider("Oswald Efficiency Factor (e)", 0.6, 0.95, 0.85)
cd0 = st.sidebar.number_input("Zero-Lift Drag (CD0)", value=0.02, step=0.005)

# Aerodynamic Calculations
cl_vec = np.linspace(0, 1.8, 100)
k = 1 / (np.pi * AR * e)
cd_vec = cd0 + k * (cl_vec**2)
ld_ratio = cl_vec / cd_vec

# Plotting
fig = go.Figure()
fig.add_trace(go.Scatter(x=cd_vec, y=cl_vec, name="Drag Polar", line=dict(color='royalblue', width=4)))
fig.update_layout(xaxis_title="Drag Coefficient (CD)", yaxis_title="Lift Coefficient (CL)", template="plotly_white")

# Dashboard UI
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)
with col2:
    st.metric("Max L/D Ratio", f"{np.max(ld_ratio):.2f}")
    st.info(f"Induced Drag Factor (k): {k:.4f}")
    st.write("**Analysis Note:** Higher Aspect Ratio significantly reduces induced drag at high lift coefficients.")