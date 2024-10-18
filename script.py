import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import joblib  # To save and load models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
from typing import List
from datetime import datetime
import os

# Directory to save/load models
MODEL_DIR = 'trained_models'
os.makedirs(MODEL_DIR, exist_ok=True)

st.set_page_config(layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ['Neural Network Explorer', 'Weights and Biases', 'Model Performance Summary', 'Other Page'])

# Function to get a unique model name
def get_unique_model_name(n_layers, n_neurons, epochs, mode):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"NN_{n_layers}layers_{n_neurons}neurons_{epochs}epochs_{mode}_{timestamp}.pkl"
    return os.path.join(MODEL_DIR, model_name)

# Neural network visualization function
def visualize_network(layers: List[int]):
    G = nx.DiGraph()
    pos = {}
    node_colors = []
    node_sizes = []

    # Loop through each layer and create nodes
    for i, layer_size in enumerate(layers):
        layer_name = "I" if i == 0 else "H" if i < len(layers) - 1 else "O"
        for j in range(layer_size):
            node_id = f"{layer_name}{i+1}_{j+1}"
            G.add_node(node_id)
            pos[node_id] = (i, -j)  # Position nodes in the graph
            node_colors.append("lightblue" if layer_name == "I" else "lightgreen" if layer_name == "O" else "lightgray")
            node_sizes.append(1000)

    # Create edges between layers
    for i in range(len(layers) - 1):
        for j in range(layers[i]):
            for k in range(layers[i+1]):
                G.add_edge(f"{'I' if i == 0 else 'H'}{i+1}_{j+1}", 
                           f"{'O' if i == len(layers) - 2 else 'H'}{i+2}_{k+1}")

    # Plot the network
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, 
            font_size=8, font_weight='bold', arrows=True)
    plt.title("Neural Network Architecture")

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# Function for calculating performance metrics
def calculate_metrics(y_true, y_pred):
    sse = np.sum((y_true - y_pred) ** 2)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mad = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'SSE': sse, 'MSE': mse, 'RMSE': rmse, 'MAD': mad, 'R²': r2}

# Function for neural network exploration page

# Function for neural network exploration page
def neural_network_explorer():
    st.title("Neural Network Explorer")

    # Sidebar mode selection
    mode = st.sidebar.selectbox("Mode", ["Classification", "Propensity"])

    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    species = iris.target_names if mode == "Classification" else ["Propensity"]

    # Sidebar controls for neural network parameters
    st.sidebar.header("Configure Neural Network")
    
    # Number of hidden layers
    n_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=5, value=1)
    
    # Number of neurons in each layer
    neurons_per_layer = []
    for i in range(n_layers):
        neurons_per_layer.append(st.sidebar.slider(f"Neurons in Hidden Layer {i+1}", min_value=1, max_value=10, value=3))

    # Number of epochs
    epochs = st.sidebar.slider("Number of Epochs", min_value=100, max_value=2000, value=1000, step=100)
    
    # Let users choose activation functions
    activation_function = st.sidebar.selectbox("Activation Function", ['relu', 'tanh', 'logistic'])

    # Split the dataset
    test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Adjust output layers based on mode
    output_nodes = 3 if mode == "Classification" else 1

    # Display dataset sample
    st.write("Sample of Iris Dataset:")
    st.write(pd.DataFrame(X, columns=iris.feature_names).head())

    # Neural Network Visualization using visualize_network function
    st.subheader("Neural Network Architecture")
    layers = [X_train.shape[1]] + neurons_per_layer + [output_nodes]
    buf = visualize_network(layers)
    st.image(buf, use_column_width=True)

    # Button to train the model
    if st.button("Train Model"):
        st.write("Training model...")

        # Train Neural Network
        hidden_layers = tuple(neurons_per_layer)  # Define hidden layer structure
        model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation_function, max_iter=epochs)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)

        if mode == "Classification":
            # Custom Confusion Matrix with "Actual" and "Predicted" labels
            st.subheader("Custom Confusion Matrix")
            conf_matrix = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
            
            # Add labels to axes
            ax.set_xticks(np.arange(len(species)))
            ax.set_yticks(np.arange(len(species)))
            ax.set_xticklabels(species)
            ax.set_yticklabels(species)
            
            # Add values inside the matrix
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='orange')

            # Add "Actual" and "Predicted" labels
            ax.set_xlabel('Predicted')
            ax.xaxis.set_label_position('top')
            ax.set_ylabel('Actual')

            st.pyplot(fig)

            # Display classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, target_names=species)
            st.text(report)

            # Initialize empty metrics for classification
            metrics = {}
            
        else:
            # Display performance metrics and plot actual vs predicted propensity scores
            st.subheader("Performance Metrics")
            metrics = calculate_metrics(y_test, y_pred)
            st.write(metrics)

            # Plot propensity scores (actual vs predicted)
            st.subheader("Propensity Scores: Actual vs Predicted")
            plt.figure(figsize=(6, 3))
            plt.plot(y_test, label="Actual Propensity", color="blue")
            plt.plot(y_pred, label="Predicted Propensity", color="orange")
            plt.xlabel("Samples")
            plt.ylabel("Propensity")
            plt.legend()
            st.pyplot()

        # Save the trained model to a unique file
        model_name = get_unique_model_name(n_layers, '_'.join(map(str, neurons_per_layer)), epochs, mode)
        joblib.dump((model, metrics, mode), model_name)
        st.success(f"Model trained and saved as {model_name}")


# Function for weights and biases page
def weights_and_biases():
    st.title("Weights and Biases Page")

    # List available models
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

    if model_files:
        selected_model = st.selectbox("Select a Model", model_files)
        model_path = os.path.join(MODEL_DIR, selected_model)

        # Load the saved model
        model, metrics, mode = joblib.load(model_path)
        st.success(f"Loaded model: {selected_model}")

        # Display weights and biases
        st.subheader("Input Layer to Hidden Layer Weights")
        for idx, layer in enumerate(model.coefs_):
            df = pd.DataFrame(layer)
            st.write(f"Layer {idx+1} Weights:")
            st.write(df)

        st.subheader("Biases for Each Layer")
        for idx, bias in enumerate(model.intercepts_):
            st.write(f"Layer {idx+1} Biases:")
            st.write(bias)
    else:
        st.error("No trained model found. Please train a model first.")

# Function for model performance summary

# Function for model performance summary
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_curve, auc

# Function for model performance summary
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# Function for model performance summary
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function for model performance summary
def model_performance_summary():
    st.title("Model Performance Summary")

    # Dropdown to filter by mode (Classification or Propensity)
    mode_filter = st.selectbox("Select Model Type", ["Classification", "Propensity"])

    # List available models filtered by mode
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pkl') and mode_filter in f]

    if model_files:
        selected_model = st.selectbox("Select a Model", model_files)
        model_path = os.path.join(MODEL_DIR, selected_model)

        # Load the selected model
        model, metrics, mode = joblib.load(model_path)

        st.subheader(f"Summary for {selected_model}")

        if mode == "Classification":
            # Performance Metrics (for Classification, we only display the confusion matrix and classification report)
            st.subheader("Classification Report")

            # Load data for predictions
            X, y = load_iris(return_X_y=True)  # Replace with your own dataset
            y_pred = model.predict(X)

            # Display Confusion Matrix
            st.subheader("Confusion Matrix")
            conf_matrix = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots()
            ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.6)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax.text(j, i, str(conf_matrix[i, j]), va='center', ha='center', color='orange')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Display Classification Report
            report = classification_report(y, y_pred, target_names=load_iris().target_names)
            st.text(report)

            # Propensity Distribution for each class (prediction probabilities)
            st.subheader("Propensity Distribution for Classes")
            y_prob = model.predict_proba(X)  # Predicted probabilities for each class
            fig, ax = plt.subplots(figsize=(8, 4))
            for i, class_name in enumerate(load_iris().target_names):
                ax.hist(y_prob[:, i], bins=10, alpha=0.6, label=class_name)
            ax.legend()
            ax.set_title("Histogram of Predicted Propensities for Each Class")
            st.pyplot(fig)

        elif mode == "Propensity":
            # Performance Metrics
            st.subheader("Performance Metrics")
            df_metrics = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            st.write(df_metrics)

            # Load data for plotting
            X, y = load_iris(return_X_y=True)  # Replace with your own dataset
            y_pred = model.predict(X)

            # Prediction Performance (Actual vs Predicted Propensity)
            st.subheader("Prediction Performance")
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(y, label="Actual Propensity", color="blue")
            ax.plot(y_pred, label="Predicted Propensity", color="orange")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Propensity")
            ax.legend()
            st.pyplot(fig)

            # Decile-wise Lift Chart
            # Decile-wise Lift Chart
            st.subheader("Decile-wise Lift Chart")
            fig, ax = plt.subplots()

            # Sort predictions and actual values based on predicted values
            sorted_indices = np.argsort(-y_pred)  # Sort in descending order of predicted values
            sorted_y = y[sorted_indices]  # Actual values sorted by predicted order

            # Handle cases where there are duplicate predicted values
            try:
                deciles = pd.qcut(sorted_y, 10, labels=False, duplicates='drop')  # Group into 10 deciles, drop duplicates if necessary
            except ValueError:
                st.warning("Not enough distinct predicted values for decile calculation.")
                deciles = pd.cut(sorted_y, bins=10, labels=False)  # Use 'cut' if qcut fails due to duplicates

            # Calculate lift for each decile
            global_mean = np.mean(y)  # Calculate the global mean of actual values
            decile_lifts = []
            for i in range(10):
                if i in deciles:
                    decile_mean = np.mean(sorted_y[deciles == i])
                    decile_lift = decile_mean / global_mean
                    decile_lifts.append(decile_lift)
                else:
                    decile_lifts.append(0)  # If decile is missing, set lift to 0

            # Plot decile-wise lift
            ax.bar(np.arange(1, 11), decile_lifts, color='blue')
            ax.set_xlabel('Decile')
            ax.set_ylabel('Decile / Global Mean')
            ax.set_title('Decile-wise Lift Chart')
            st.pyplot(fig)


            # Lift Chart - Ideal vs Predicted
            st.subheader("Lift Chart")
            fig, ax = plt.subplots()

            # Sort y (actuals) and y_pred (predicted probabilities)
            sorted_indices = np.argsort(-y_pred)  # Sort by predicted values in descending order
            sorted_y = y[sorted_indices]  # Sort actuals by predicted order

            # Calculate cumulative for predicted and random
            cumulative_predicted = np.cumsum(sorted_y) / np.sum(sorted_y)  # Cumulative predicted lift
            cumulative_random = np.linspace(0, 1, len(sorted_y))  # Baseline (random) cumulative lift

            # Plot the lift chart
            ax.plot(np.arange(len(sorted_y)), cumulative_predicted, label="Cumulative actual when sorted by predicted values", color="blue")
            ax.plot(np.arange(len(sorted_y)), cumulative_random, label="Cumulative actual using average (Random)", color="red")

            # Chart details
            ax.set_xlabel("# Cases")
            ax.set_ylabel("Cumulative")
            ax.legend(loc="lower right")
            ax.set_title("Lift Chart: Ideal vs Predicted")
            st.pyplot(fig)



            # ROC Curve
            st.subheader("ROC Curve")
            if len(np.unique(y)) == 2:  # Ensure the task is binary
                fpr, tpr, _ = roc_curve(y, y_pred)
                roc_auc = auc(fpr, tpr)

                fig, ax = plt.subplots()
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("ROC curve is only applicable for binary outcomes.")
    else:
        st.error("No models found for the selected type. Please train a model first.")

# Placeholder for other pages
def other_page():
    st.title("Other Page")
    st.write("This page is under construction.")

# Map pages to functions
page_functions = {
    'Neural Network Explorer': neural_network_explorer,
    'Weights and Biases': weights_and_biases,
    'Model Performance Summary': model_performance_summary,
    'Other Page': other_page
}

# Display the selected page
if page in page_functions:
    page_functions[page]()
else:
    st.error("Page not found!")
