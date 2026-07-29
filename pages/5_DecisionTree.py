import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import plot_tree
from sklearn.tree import export_text

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from utils.loader import load_data
from utils.preprocess import preprocess
from utils.train import train_tree
from utils.charts import show_confusion

st.set_page_config(

    page_title="Decision Tree",

    page_icon="🌳",

    layout="wide"

)

st.title("🌳 Decision Tree")

df = load_data()

X_train,X_test,y_train,y_test,preprocessor = preprocess(df)