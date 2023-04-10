import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from PIL import Image

# load data
gold_data = pd.read_csv('gld_price_data.csv')

# split into X and Y
X = gold_data.drop(columns=['Date', 'GLD'], axis=1)
Y = gold_data['GLD']
print(X.shape, "\n", Y.shape)

# split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=2)

print(X_train.shape, X_test.shape)

reg = RandomForestRegressor()

reg = reg.fit(X_train, Y_train)

pred = reg.predict(X_test)
score = r2_score(Y_test, pred)

# webapp

st.title("Gold Price Prediction")
st.write("This is a web app to predict the price of gold")

image = Image.open('gold.webp')
st.image(image, width=200, caption='Gold', use_column_width=True)

st.subheader("Enter the data to predict the price of gold")

# user input


# def user_input():
#     Open = st.number_input("Open")
#     High = st.number_input("High")
#     Low = st.number_input("Low")
#     Close = st.number_input("Close")
#     Volume = st.number_input("Volume")
#     data = {'Open': Open,
#             'High': High,
#             'Low': Low,
#             'Close': Close,
#             'Volume': Volume}
#     features = pd.DataFrame(data, index=[0])
#     return features


# df = user_input()


# st.subheader("User Input")
# st.write(df)

st.subheader("Prediction")
st.write(gold_data)

st.subheader("Accuracy")
st.write(score)
