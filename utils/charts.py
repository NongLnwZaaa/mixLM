import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def show_confusion(cm):

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        fmt="d",
        ax=ax
    )

    ax.set_xlabel("Prediction")

    ax.set_ylabel("Actual")

    st.pyplot(fig)