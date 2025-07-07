import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt

# Load model
model = joblib.load("iris_model.pkl")
iris = load_iris()

st.title("🌸 Iris Flower Classifier")
st.write("Input flower features to classify its species.")

# Sidebar input
st.sidebar.header("Enter Flower Measurements")
sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sl, sw, pl, pw]])
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)

st.subheader("Prediction")
st.write(f"Predicted Species: **{iris.target_names[prediction[0]]}**")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(probability, columns=iris.target_names)
st.bar_chart(proba_df.T)

st.subheader("Visualize Input")
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

fig, ax = plt.subplots()
sns.scatterplot(
    x='petal length (cm)', y='petal width (cm)',
    hue='target', palette='deep', data=df, legend=False
)
ax.scatter(pl, pw, color='black', s=100, label="Your Input")
st.pyplot(fig)
