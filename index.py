# pylint: disable=import-error
import os
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import seaborn as sns
import plotly.express as px
from plotly import tools
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit import caching
import pandas as pd
pd.plotting.register_matplotlib_converters()

st.set_option('deprecation.showPyplotGlobalUse', False)

filepath = "./assets/data.csv"
raw_data = pd.read_csv(filepath)

extracted_data = raw_data.loc[raw_data['Country Name'] == 'Malaysia']
cleaned_data = extracted_data.dropna(axis=1).drop(
    columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])
data = cleaned_data.melt(var_name='Year', value_name='Value').sort_values(
    ['Year']).reset_index(drop=True)
data['Year Index'] = data.index + 1

model = tf.keras.Sequential([
    layers.Dense(units=1)
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

model.fit(
    data['Year Index'], data['Value'],
    epochs=50
)


def plot_value(x, y):
    plt.scatter(data['Year Index'], data['Value'], label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Year Index')
    plt.ylabel('Value')
    plt.title('Carbon Emissions')
    plt.legend()


x = tf.linspace(1, 70, 2)
y = model.predict(x)

option = st.sidebar.selectbox('', ['Project Overview', 'Project Demo'])

if option == "Project Overview":
    st.title('Project Title')
    st.write('Project Description')

    st.subheader('Problem Statement')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')
    st.write('')

    plot_value(x, y)
    st.pyplot(clear_figure=False)

    st.subheader('Solution')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')

    st.subheader('Future Improvements')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')

    st.subheader('Our Team')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')

    st.subheader('Appendix')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')

elif option == "Project Demo":
    st.title('Project Demo')
