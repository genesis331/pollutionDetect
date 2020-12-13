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

option = st.sidebar.selectbox(
    '', ['Project Overview', 'Project Demo', 'Linear Regression Prediction'])

if option == "Project Overview":
    from PIL import Image

    st.title('Project Title')
    st.write('Deep Learning for Traffic Analysis and Climate Control')

    st.header('Problem Statement')
    st.write('Carbon emissions from car exhaust gases contain a great number of chemical substances that are detrimental not only to the human body, but also to environmental health. In a country with high car ownership like Malaysia, many environmentally harmful gases and substances are released into the surroundings on a daily basis. In the long term, this phenomenon leads to cases of climate change, particularly global warming. As seen in the graph below, the rate of carbon emission has seen a steady increase in the past 50 years.')
    st.write('')

    img = Image.open("assets/graph.png")
    st.image(img, width=600)

    st.header('Solution')
    st.write('1) A clear photo of incoming traffic is taken to monitor at regular intervals on Road X, preferably during the red light when cars are stationary so as to ease the process of analysis and object detection.')
    st.write('2) Traffic analysis is carried out by counting the number of vehicles at the time of the monitoring using FDK object_detection.')

    st.header('Future Improvements')
    st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam eu laoreet lacus, non euismod quam. Vivamus ac erat ut magna pretium iaculis. Pellentesque at lorem augue. Sed non orci non nisi suscipit tincidunt a nec tellus. Sed feugiat turpis nec felis accumsan, et malesuada elit consectetur. Nullam blandit luctus erat mattis ultrices. Interdum et malesuada fames ac ante ipsum primis in faucibus. Vestibulum efficitur gravida velit, in lobortis sapien lacinia eget. Integer sagittis est turpis, eget aliquam risus pulvinar nec.')

    st.header('Our Team')
    st.write('')
    col1, col2, col3 = st.beta_columns(3)
    col1.image(Image.open("assets/cheahzixu.jpg"),width=150)
    col1.subheader('Cheah Zixu')
    col1.write('@Xephius')
    col1.write('GitHub: @genesis331')
    col1.write('IG: @zixucheah331')
    col2.image(Image.open("assets/choochenzhung.jpg"),width=150)
    col2.subheader('Choo Chen Zhung')
    col2.write('@Xectroz')
    col2.write('GitHub: @Deadsteration')
    col2.write('IG: @czhung0701')
    col3.image(Image.open("assets/hoshuyi.jpg"),width=150)
    col3.subheader('Ho Shu Yi')
    col3.write('@Xytrix')
    col3.write('GitHub: @Xytrix1103')
    col3.write('IG: @shuyi_ho03')

    st.header('Appendix')
    st.text('https://www.geography.org.uk/teaching-resources/singapore-malaysia/Can-Malaysia-do-anything-about-its-air-pollution#:~:text=The%20first%20is%20air%20pollution,in%20all%20its%20major%20cities.')
    st.text('https://www.ucsusa.org/resources/cars-trucks-buses-and-air-pollution')
    st.text('https://www.researchgate.net/publication/317304216_Verification_Relationship_between_Vehicle_Data_and_Air_Pollution_Index_Using_Muti-linear_Regression_Modeling')
    st.text('https://www.researchgate.net/publication/286197080_Air_pollution_study_of_vehicles_emission_in_high_volume_traffic_Selangor_Malaysia_as_a_case_study')
    st.text('https://www.titlemax.com/resources/the-effect-of-your-cars-carbon-emission/')

elif option == "Project Demo":
    st.title('Project Demo')
    st.write('')

    left_column, right_column = st.beta_columns(2)
    car_limit = left_column.number_input('Cars',min_value=0,value=0)
    bus_limit = left_column.number_input('Buses',min_value=0,value=0)
    truck_limit = left_column.number_input('Trucks',min_value=0,value=0)
    motorcycle_limit = left_column.number_input('Motorcycles',min_value=0,value=0)
    right_column.header('Total vehicle limit: ' + str(car_limit + bus_limit + truck_limit + motorcycle_limit))
    st.write('')

    st.echo()
    with st.echo():
        import torch
        from src.core.detect import Detector
        from src.core.utils import utils
        from PIL import Image
        import cv2

    st.echo()
    with st.echo():
        det = Detector(name="DemoDet")

    st.echo()
    with st.echo():
        img = Image.open("assets/highway.jpg")
        st.image(img, width=700)

    st.echo()
    with st.echo():
        img_cv = utils.pil_to_cv2(img)
        output = det.predict(img_cv)
        out_img = det.visualize(img_cv, output, figsize=(18, 18))
        cv2.imwrite('tempImage.jpg', out_img)
        st.image('tempImage.jpg', width=700)

    objects = getattr(output['instances'],'pred_classes')
    list = objects.tolist()
    dict = {
        "cars": 0,
        "bus": 0,
        "truck": 0,
        "motorcycle": 0,
        "total": 0
    }
    dict["cars"] += list.count(2)
    dict["total"] += list.count(2)
    dict["bus"] += list.count(5)
    dict["total"] += list.count(5)
    dict["truck"] += list.count(7)
    dict["total"] += list.count(7)
    dict["motorcycle"] += list.count(3)
    dict["total"] += list.count(3)
    st.table(pd.DataFrame(dict.items(),columns=['Vehicle Type','Value']))
    st.write('')

    imgTick = Image.open("assets/check-circle.png")
    imgX = Image.open("assets/alert-circle.png")

    left_column1, right_column1 = st.beta_columns(2)
    left_column1.subheader('Car:')
    if car_limit == 0:
        left_column1.image(imgTick,width=30)
    elif car_limit >= dict["cars"]:
        left_column1.image(imgTick,width=30)
    else:
        left_column1.image(imgX,width=30)

    left_column1.subheader('Bus:')
    if bus_limit == 0:
        left_column1.image(imgTick,width=30)
    elif bus_limit >= dict["bus"]:
        left_column1.image(imgTick,width=30)
    else:
        left_column1.image(imgX,width=30)

    left_column1.subheader('Truck:')
    if truck_limit == 0:
        left_column1.image(imgTick,width=30)
    elif truck_limit >= dict["truck"]:
        left_column1.image(imgTick,width=30)
    else:
        left_column1.image(imgX,width=30)

    left_column1.subheader('Motorcycle:')
    if motorcycle_limit == 0:
        left_column1.image(imgTick,width=30)
    elif motorcycle_limit >= dict["motorcycle"]:
        left_column1.image(imgTick,width=30)
    else:
        left_column1.image(imgX,width=30)

    right_column1.subheader('Number of vehicles in total:')
    if (car_limit + bus_limit + truck_limit + motorcycle_limit) == 0:
        right_column1.image(imgTick,width=60)
    elif (car_limit + bus_limit + truck_limit + motorcycle_limit) >= len(list):
        right_column1.image(imgTick,width=60)
    else:
        right_column1.image(imgX,width=60)

elif option == "Linear Regression Prediction":
    st.title('Extra stuff')
    st.write('A simple prediction of future carbon emissions')
    st.write('')

    st.echo()
    with st.echo():
        # Read downloaded dataset from Kaggle
        filepath = "./assets/data.csv"
        raw_data = pd.read_csv(filepath)

        # Remove unwanted contents and restructure the data
        extracted_data = raw_data.loc[raw_data['Country Name'] == 'Malaysia']
        cleaned_data = extracted_data.dropna(axis=1).drop(
            columns=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'])
        data = cleaned_data.melt(var_name='Year', value_name='Value').sort_values(
            ['Year']).reset_index(drop=True)
        data['Year Index'] = data.index + 1
    
    st.echo()
    with st.echo():
        # Define a model
        model = tf.keras.Sequential([
            layers.Dense(units=1)
        ])

        # Compile a model
        model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error'
        )

        # Fit training data to model
        history = model.fit(
            data['Year Index'], data['Value'],
            epochs=50
        )

        # Store the logs
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

    st.echo()
    with st.echo():
        # Make prediction
        x = tf.linspace(1, 70, 2)
        y = model.predict(x)

    st.echo()
    with st.echo():
        # Plot a loss graph
        plt.plot(history.history['loss'], label='loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        st.pyplot(clear_figure=True)

    st.echo()
    with st.echo():
        # Scatterplot of data and adding regression line
        plt.scatter(data['Year Index'], data['Value'], label='Data')
        plt.plot(x, y, color='m', label='Predictions')
        plt.xlabel('Year Index')
        plt.ylabel('Value')
        plt.title('Carbon Emissions')
        plt.legend()
        st.pyplot(clear_figure=True)
