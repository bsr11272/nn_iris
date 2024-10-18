import streamlit as st
import numpy as np
import pandas as pd

# Function to extract weights and biases from the neural network
def show_weights_biases(weights, biases):
    st.title("Neural Network Weights for Classification")

    # Display weights for input layer to hidden layer
    st.subheader("Neuron Weights: Input Layer - Hidden Layer 1")
    columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Bias']
    df_weights_input_hidden = pd.DataFrame(np.hstack((weights[0], biases[0].reshape(-1, 1))), columns=columns)
    df_weights_input_hidden.index = [f"Neuron {i+1}" for i in range(df_weights_input_hidden.shape[0])]
    st.write(df_weights_input_hidden.style.format("{:.6f}").highlight_max(axis=1))

    # Display weights for hidden layer to output layer
    st.subheader("Neuron Weights: Hidden Layer 1 - Output Layer")
    output_columns = [f"Neuron {i+1}" for i in range(weights[1].shape[1])] + ['Bias']
    df_weights_hidden_output = pd.DataFrame(np.hstack((weights[1].T, biases[1].reshape(-1, 1))), columns=output_columns)
    df_weights_hidden_output.index = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    st.write(df_weights_hidden_output.style.format("{:.6f}").highlight_max(axis=1))
