import streamlit as st
from streamlit import caching
import os

option = st.sidebar.selectbox('',['Project Overview','Project Demo'])

if option == "Project Overview":
    st.title('Project Title')
    st.write('Project Description')

elif option == "Project Demo":
    st.title('Project Demo')