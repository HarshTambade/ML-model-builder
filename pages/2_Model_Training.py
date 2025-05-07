"""
ALPHA - End-to-End Machine Learning Platform
Model Training Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from pathlib import Path
import os
import json

# Import utility modules
from utils.config import get_available_datasets, DEFAULT_MODELS
from utils.data import get_dataset_by_id, split_dataset
from utils.models import (
    get_sklearn_model, train_sklearn_model, evaluate_classification_model,
    evaluate_regression_model, evaluate_clustering_model, save_model
)
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)
from utils.imports import is_package_available, logger, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Model Training")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Model Training",
    description="Train, evaluate, and save machine learning models",
    icon="🔬"
)

# Create tabs for the model training workflow
training_tabs = create_tab_panels("Select Dataset", "Configure Model", "Training Results")

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for model training. Please install pandas.')
    st.stop()
if not is_package_available('matplotlib'):
    st.warning('Matplotlib is not available. Some visualizations may not work.')
if not is_package_available('seaborn'):
    st.warning('Seaborn is not available. Some visualizations may not work.')

# Tab 1: Select Dataset
with training_tabs[0]:
    st.markdown("### Select a Dataset for Training")
    
    # Add option to use existing dataset or upload new one
    dataset_source = st.radio(
        "Dataset Source",
        ["Use Existing Dataset", "Upload Custom CSV File"],
        help="Choose whether to use an existing dataset or upload a new one"
    )
    
    if dataset_source == "Upload Custom CSV File":
        st.markdown("### Upload your CSV file for model training")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=["csv"]
        )
        
        # If a file is uploaded
        if uploaded_file is not None:
            try:
                # Show file details
                st.markdown(f"**File Details**:")
                col1, col2 = st.columns(2)
                col1.markdown(f"**Name:** {uploaded_file.name}")
                col2.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
                
                # Load the data
                try:
                    df = pd.read_csv(uploaded_file)
                    # Fix DataFrame for display
                    df = fix_dataframe_dtypes(df)
                    # Validate DataFrame before display
                    is_valid, msg, problematic = validate_dataframe_for_streamlit(df)
                    if not is_valid:
                        st.error(f"Cannot display DataFrame: {msg}")
                    else:
                        st.success(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
                        
                        # Display data preview
                        st.markdown("#### Data Preview")
                        st.dataframe(df.head(10))
                        
                        # Basic dataset info
                        st.markdown("#### Dataset Information")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Rows", df.shape[0])
                        col2.metric("Columns", df.shape[1])
                        col3.metric("Size (KB)", round(df.memory_usage(deep=True).sum() / 1024, 2))
                        
                        # Target column selection
                        st.markdown("### Select Target Variable")
                        target_column = st.selectbox(
                            "Select the target column for prediction",
                            options=df.columns.tolist(),
                            index=len(df.columns)-1  # Default to last column
                        )
                        
                        # Create temporary dataset info structure
                        temp_dataset_info = {
                            "data": df,
                            "metadata": {
                                "name": os.path.splitext(uploaded_file.name)[0],
                                "description": "Temporary uploaded dataset",
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "type": "uploaded_temp"
                            }
                        }
                        
                        # Store uploaded dataset in session state
                        if st.button("Use This Dataset"):
                            st.session_state["selected_dataset"] = temp_dataset_info
                            st.session_state["target_column"] = target_column
                            
                            # Switch to the next tab
                            st.success(f"Uploaded dataset selected for training!")
                            st.info("Now configure your model in the next tab.")
                
                except Exception as e:
                    st.error(f"Error loading dataset: {str(e)}")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    else:
        # Get available datasets
        datasets = get_available_datasets()
        
        if not datasets:
            st.warning("No datasets available. Please upload or generate a dataset in the Data Management module.")
            
            # Add a button to navigate to the Data Management page
            if st.button("Go to Data Management"):
                st.switch_page("pages/1_Data_Management.py")
            
            # Display instructions on how to add data
            with st.expander("How to add data?"):
                st.markdown("""
                ### Adding Data to the Platform
                
                1. **Upload your own dataset**:
                   - Go to the Data Management page
                   - Use the 'Upload Dataset' tab
                   - Select a file to upload (CSV, Excel, JSON, etc.)
                
                2. **Generate synthetic data**:
                   - Go to the Data Management page
                   - Use the 'Generate Dataset' tab
                   - Configure the parameters and generate a dataset
                """)
        else:
            # Dataset selection
            dataset_names = [dataset["name"] for dataset in datasets]
            selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
            
            # Find the selected dataset
            selected_dataset = next(
                (dataset for dataset in datasets if dataset["name"] == selected_dataset_name),
                None
            )
            
            if selected_dataset:
                # Extract dataset path to get the ID
                dataset_path = Path(selected_dataset["path"])
                dataset_id = dataset_path.name
                
                # Get the dataset
                dataset_info = get_dataset_by_id(dataset_id)
                
                if dataset_info and dataset_info["data"] is not None:
                    df = dataset_info["data"]
                    # Fix DataFrame for display
                    df = fix_dataframe_dtypes(df)
                    # Validate DataFrame before display
                    is_valid, msg, problematic = validate_dataframe_for_streamlit(df)
                    if not is_valid:
                        st.error(f"Cannot display DataFrame: {msg}")
                    else:
                        # Display dataset preview
                        st.markdown("#### Dataset Preview")
                        st.dataframe(df.head())
                        
                        # Basic dataset info
                        st.markdown("#### Dataset Information")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Rows", df.shape[0])
                        col2.metric("Columns", df.shape[1])
                        col3.metric("Size (KB)", round(df.memory_usage(deep=True).sum() / 1024, 2))
                        
                        # Target column selection
                        st.markdown("### Select Target Variable")
                        target_column = st.selectbox(
                            "Select the target column for prediction",
                            options=df.columns.tolist(),
                            index=len(df.columns)-1  # Default to last column
                        )
                        
                        # Store selected dataset in session state
                        if st.button("Use This Dataset"):
                            st.session_state["selected_dataset"] = dataset_info
                            st.session_state["target_column"] = target_column
                            
                            # Switch to the next tab
                            st.success(f"Dataset '{selected_dataset_name}' selected for training!")
                            st.info("Now configure your model in the next tab.")
                else:
                    st.error("Error loading dataset data. The dataset might be corrupted.")
                    
                    # Add a button to navigate to the Data Management page
                    if st.button("Go to Data Management"):
                        st.switch_page("pages/1_Data_Management.py")

# Tab 2: Configure Model
with training_tabs[1]:
    st.markdown("### Configure Your Model")
    
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset in the previous tab.")
    else:
        dataset_info = st.session_state["selected_dataset"]
        df = dataset_info["data"]
        target_column = st.session_state["target_column"]
        
        # Display selected dataset and target
        st.markdown(f"**Selected Dataset:** {dataset_info['metadata']['name']}")
        st.markdown(f"**Target Column:** {target_column}")
        
        # Determine the problem type based on target column
        y = df[target_column]
        is_categorical = y.dtype == "object" or y.dtype == "category" or len(np.unique(y)) < 10
        
        # Set default problem type based on data
        default_problem_type = "classification" if is_categorical else "regression"
        
        # Model configuration
        st.markdown("#### Model Configuration")
        
        col1, col2 = st.columns(2)
        
        # Problem type
        problem_type = col1.selectbox(
            "Problem Type",
            ["classification", "regression", "clustering"],
            index=0 if default_problem_type == "classification" else 1
        )
        
        # Model selection based on problem type
        model_options = DEFAULT_MODELS.get(problem_type, [""])
        selected_model = col2.selectbox("Select Model", model_options)
        
        # Data splitting
        st.markdown("#### Data Splitting")
        
        col1, col2, col3 = st.columns(3)
        test_size = col1.slider("Test Size (%)", 10, 40, 20) / 100
        val_size = col2.slider("Validation Size (%)", 10, 40, 20) / 100
        random_state = col3.number_input("Random State", 0, 100, 42)
        
        # Model hyperparameters
        st.markdown("#### Model Hyperparameters")
        
        # Different hyperparameters based on selected model
        hyperparams = {}
        
        if problem_type == "classification":
            if selected_model == "LogisticRegression":
                col1, col2 = st.columns(2)
                C = col1.number_input("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                max_iter = col2.number_input("Max Iterations", 100, 1000, 100, 100)
                hyperparams = {"C": C, "max_iter": max_iter}
            
            elif selected_model == "RandomForest":
                col1, col2, col3 = st.columns(3)
                n_estimators = col1.number_input("Number of Trees", 10, 500, 100, 10)
                max_depth = col2.number_input("Max Depth", 1, 100, 10, 1)
                min_samples_split = col3.number_input("Min Samples Split", 2, 20, 2, 1)
                hyperparams = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }
            
            elif selected_model == "SVC":
                col1, col2 = st.columns(2)
                C = col1.number_input("Regularization (C)", 0.01, 10.0, 1.0, 0.01)
                kernel = col2.selectbox("Kernel", ["linear", "rbf", "poly"])
                hyperparams = {"C": C, "kernel": kernel}
            
            elif selected_model == "KNeighbors":
                col1, col2 = st.columns(2)
                n_neighbors = col1.number_input("K Neighbors", 1, 50, 5, 1)
                weights = col2.selectbox("Weights", ["uniform", "distance"])
                hyperparams = {"n_neighbors": n_neighbors, "weights": weights}
            
            elif selected_model == "DecisionTree":
                col1, col2 = st.columns(2)
                max_depth = col1.number_input("Max Depth", 1, 100, 10, 1)
                min_samples_split = col2.number_input("Min Samples Split", 2, 20, 2, 1)
                hyperparams = {"max_depth": max_depth, "min_samples_split": min_samples_split}
        
        elif problem_type == "regression":
            if selected_model == "LinearRegression":
                st.info("LinearRegression has no hyperparameters to tune.")
                hyperparams = {}
            
            elif selected_model == "Ridge":
                col1, col2 = st.columns(2)
                alpha = col1.number_input("Alpha", 0.01, 10.0, 1.0, 0.01)
                max_iter = col2.number_input("Max Iterations", 100, 1000, 100, 100)
                hyperparams = {"alpha": alpha, "max_iter": max_iter}
            
            elif selected_model == "Lasso":
                col1, col2 = st.columns(2)
                alpha = col1.number_input("Alpha", 0.01, 10.0, 1.0, 0.01)
                max_iter = col2.number_input("Max Iterations", 100, 1000, 100, 100)
                hyperparams = {"alpha": alpha, "max_iter": max_iter}
            
            elif selected_model == "RandomForestRegressor":
                col1, col2, col3 = st.columns(3)
                n_estimators = col1.number_input("Number of Trees", 10, 500, 100, 10)
                max_depth = col2.number_input("Max Depth", 1, 100, 10, 1)
                min_samples_split = col3.number_input("Min Samples Split", 2, 20, 2, 1)
                hyperparams = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split
                }
        
        elif problem_type == "clustering":
            if selected_model == "KMeans":
                col1, col2 = st.columns(2)
                n_clusters = col1.number_input("Number of Clusters", 2, 20, 3, 1)
                max_iter = col2.number_input("Max Iterations", 100, 1000, 300, 100)
                hyperparams = {"n_clusters": n_clusters, "max_iter": max_iter}
            
            elif selected_model == "DBSCAN":
                col1, col2 = st.columns(2)
                eps = col1.number_input("Epsilon", 0.01, 5.0, 0.5, 0.01)
                min_samples = col2.number_input("Min Samples", 1, 20, 5, 1)
                hyperparams = {"eps": eps, "min_samples": min_samples}
        
        # Training button
        if st.button("Train Model"):
            # Store training configuration in session state
            st.session_state["problem_type"] = problem_type
            st.session_state["selected_model"] = selected_model
            st.session_state["hyperparams"] = hyperparams
            st.session_state["test_size"] = test_size
            st.session_state["val_size"] = val_size
            st.session_state["random_state"] = random_state
            
            # Start training
            with st.spinner("Training model..."):
                # Split the data
                if problem_type != "clustering":
                    # Prepare data
                    X = df.drop(columns=[target_column])
                    y = df[target_column]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Create model instance
                    model = get_sklearn_model(
                        model_name=selected_model,
                        **hyperparams
                    )
                    
                    # Train the model
                    start_time = time.time()
                    model = train_sklearn_model(model, X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    if problem_type == "classification":
                        eval_results = evaluate_classification_model(model, X_test, y_test)
                    elif problem_type == "regression":
                        eval_results = evaluate_regression_model(model, X_test, y_test)
                    
                    # Store results in session state
                    st.session_state["model"] = model
                    st.session_state["evaluation"] = eval_results
                    st.session_state["X_train"] = X_train
                    st.session_state["X_test"] = X_test
                    st.session_state["y_train"] = y_train
                    st.session_state["y_test"] = y_test
                    st.session_state["training_time"] = training_time
                
                else:  # Clustering
                    # For clustering, we don't use target column
                    X = df.drop(columns=[target_column]) if target_column in df.columns else df
                    
                    # Create model instance
                    model = get_sklearn_model(
                        model_name=selected_model,
                        **hyperparams
                    )
                    
                    # Train the model
                    start_time = time.time()
                    model = train_sklearn_model(model, X)
                    training_time = time.time() - start_time
                    
                    # Evaluate model
                    eval_results = evaluate_clustering_model(model, X)
                    
                    # Store results in session state
                    st.session_state["model"] = model
                    st.session_state["evaluation"] = eval_results
                    st.session_state["X"] = X
                    st.session_state["training_time"] = training_time
            
            st.success("Model training completed! Check the Training Results tab.")

# Tab 3: Training Results
with training_tabs[2]:
    st.markdown("### Model Training Results")
    
    if "selected_dataset" not in st.session_state:
        st.warning("Please select a dataset and configure a model first.")
    elif "model" not in st.session_state:
        st.info("No model has been trained yet. Configure and train your model in the previous tabs.")
    else:
        # Get the trained model and results
        model = st.session_state.get("model")
        evaluation = st.session_state.get("evaluation", {})
        problem_type = st.session_state.get("problem_type", "")
        selected_model = st.session_state.get("selected_model", "")
        training_time = st.session_state.get("training_time", 0)
        
        # Display model info
        st.markdown(f"### Trained Model: {selected_model}")
        st.markdown(f"**Problem Type:** {problem_type.capitalize()}")
        
        # Display training details
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Time", f"{training_time:.2f}s")
        
        if problem_type != "clustering":
            X_train = st.session_state.get("X_train")
            X_test = st.session_state.get("X_test")
            col2.metric("Train Size", X_train.shape[0] if X_train is not None else 0)
            col3.metric("Test Size", X_test.shape[0] if X_test is not None else 0)
        else:
            X = st.session_state.get("X")
            col2.metric("Data Size", X.shape[0] if X is not None else 0)
            col3.metric("Features", X.shape[1] if X is not None else 0)
        
        # Display performance metrics
        st.markdown("### Model Performance")
        
        if problem_type == "classification":
            if "metrics" in evaluation:
                metrics = evaluation.get("metrics", {})
                metric_cols = st.columns(4)
                
                metric_cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                metric_cols[1].metric("Precision", f"{metrics.get('precision', 0):.4f}")
                metric_cols[2].metric("Recall", f"{metrics.get('recall', 0):.4f}")
                metric_cols[3].metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                
                # Display confusion matrix
                if "confusion_matrix" in evaluation:
                    st.markdown("#### Confusion Matrix")
                    cm = evaluation["confusion_matrix"]
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
                
                # Display ROC curve
                if "roc_data" in evaluation and evaluation["roc_data"] is not None:
                    st.markdown("#### ROC Curve")
                    roc_data = evaluation["roc_data"]
                    fpr = roc_data["fpr"]
                    tpr = roc_data["tpr"]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f"AUC = {metrics.get('auc', 0):.4f}")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
            else:
                st.warning("No evaluation metrics available for this model.")
        
        elif problem_type == "regression":
            if "metrics" in evaluation:
                metrics = evaluation.get("metrics", {})
                metric_cols = st.columns(4)
                
                metric_cols[0].metric("R² Score", f"{metrics.get('r2', 0):.4f}")
                metric_cols[1].metric("MAE", f"{metrics.get('mae', 0):.4f}")
                metric_cols[2].metric("MSE", f"{metrics.get('mse', 0):.4f}")
                metric_cols[3].metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                
                # Display residuals plot
                if "residuals_data" in evaluation:
                    st.markdown("#### Residuals Plot")
                    residuals_data = evaluation["residuals_data"]
                    y_test = residuals_data["y_test"]
                    y_pred = residuals_data["y_pred"]
                    residuals = residuals_data["residuals"]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_pred, residuals)
                    ax.axhline(y=0, color="r", linestyle="-")
                    ax.set_xlabel("Predicted Values")
                    ax.set_ylabel("Residuals")
                    ax.set_title("Residuals Plot")
                    st.pyplot(fig)
                    
                    # Display actual vs predicted
                    st.markdown("#### Actual vs Predicted")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, y_pred)
                    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Actual vs Predicted")
                    st.pyplot(fig)
            else:
                st.warning("No evaluation metrics available for this model.")
        
        elif problem_type == "clustering":
            if "metrics" in evaluation:
                metrics = evaluation.get("metrics", {})
                metric_cols = st.columns(3)
                
                if "silhouette" in metrics and metrics["silhouette"] is not None:
                    metric_cols[0].metric("Silhouette Score", f"{metrics.get('silhouette', 0):.4f}")
                if "inertia" in metrics:
                    metric_cols[1].metric("Inertia", f"{metrics.get('inertia', 0):.4f}")
                if "n_clusters" in metrics:
                    metric_cols[2].metric("Clusters", metrics.get("n_clusters", 0))
                
                # If cluster distribution is available, display it
                if "cluster_distribution" in evaluation:
                    st.markdown("#### Cluster Distribution")
                    cluster_dist = evaluation["cluster_distribution"]
                    
                    # Convert to DataFrame for display
                    dist_df = pd.DataFrame({
                        "Cluster": list(cluster_dist.keys()),
                        "Count": list(cluster_dist.values())
                    })
                    
                    st.dataframe(dist_df)
                    
                    # Plot cluster distribution
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(list(cluster_dist.keys()), list(cluster_dist.values()))
                    ax.set_xlabel("Cluster")
                    ax.set_ylabel("Count")
                    ax.set_title("Cluster Distribution")
                    st.pyplot(fig)
                
                # If we have 2D data, visualize clusters
                if "X" in st.session_state and "labels" in evaluation:
                    X = st.session_state["X"]
                    labels = evaluation["labels"]
                    
                    if X.shape[1] >= 2:
                        st.markdown("#### Cluster Visualization (First 2 Dimensions)")
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", alpha=0.6)
                        
                        # Add centroids if KMeans
                        if hasattr(model, "cluster_centers_"):
                            centroids = model.cluster_centers_
                            ax.scatter(centroids[:, 0], centroids[:, 1], s=300, c="red", marker="X")
                        
                        ax.set_xlabel(f"Feature 1: {X.columns[0]}")
                        ax.set_ylabel(f"Feature 2: {X.columns[1]}")
                        ax.set_title("Cluster Visualization")
                        plt.colorbar(scatter, ax=ax, label="Cluster")
                        st.pyplot(fig)
            else:
                st.warning("No evaluation metrics available for this model.")
        
        # Save model section
        st.markdown("### Save Model")
        col1, col2 = st.columns(2)
        with col1:
            model_name = st.text_input("Model Name", value=f"{selected_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        with col2:
            model_tags = st.text_input("Tags (comma separated)", placeholder="regression, timeseries, stock")
        
        model_description = st.text_area("Model Description", 
            value=f"{problem_type.capitalize()} model trained on {st.session_state['selected_dataset']['metadata']['name']} dataset.")
        
        if st.button("Save Model"):
            if not model_name:
                st.error("Please enter a model name.")
            else:
                with st.spinner("Saving model..."):
                    try:
                        # Parse tags
                        tags = [tag.strip() for tag in model_tags.split(",")] if model_tags else []
                        
                        # Save the model
                        result = save_model(
                            model=model,
                            model_name=model_name,
                            model_type=selected_model,
                            description=model_description,
                            performance=evaluation.get("metrics", {}),
                            dataset_name=st.session_state["selected_dataset"]["metadata"]["name"],
                            problem_type=problem_type,
                            parameters=st.session_state.get("hyperparams", {}),
                            metadata={
                                "tags": tags,
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "dataset_id": st.session_state["selected_dataset"]["metadata"].get("id", "")
                            }
                        )
                        
                        st.success(f"Model saved successfully! Model ID: {result['id']}")
                        st.markdown(f"You can now use this model in the Deployment module.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                    
        # Add option to download model report
        st.markdown("### Export Model Report")
        report_formats = st.multiselect(
            "Select report formats",
            options=["Text", "HTML", "JSON"],
            default=["HTML"]
        )
        
        if st.button("Generate Model Report"):
            # Create report content
            report_content = f"""
            # Model Training Report
            
            ## Model Information
            - **Model Name:** {model_name}
            - **Model Type:** {selected_model}
            - **Problem Type:** {problem_type.capitalize()}
            - **Dataset:** {st.session_state["selected_dataset"]["metadata"]["name"]}
            - **Training Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            ## Performance Metrics
            """
            
            # Add metrics based on problem type
            metrics = evaluation.get("metrics", {})
            if problem_type == "classification":
                report_content += f"""
                - **Accuracy:** {metrics.get('accuracy', 0):.4f}
                - **Precision:** {metrics.get('precision', 0):.4f}
                - **Recall:** {metrics.get('recall', 0):.4f}
                - **F1 Score:** {metrics.get('f1', 0):.4f}
                """
            elif problem_type == "regression":
                report_content += f"""
                - **R² Score:** {metrics.get('r2', 0):.4f}
                - **MAE:** {metrics.get('mae', 0):.4f}
                - **MSE:** {metrics.get('mse', 0):.4f}
                - **RMSE:** {metrics.get('rmse', 0):.4f}
                """
            elif problem_type == "clustering":
                if "silhouette" in metrics and metrics["silhouette"] is not None:
                    report_content += f"- **Silhouette Score:** {metrics.get('silhouette', 0):.4f}\n"
                if "inertia" in metrics:
                    report_content += f"- **Inertia:** {metrics.get('inertia', 0):.4f}\n"
                if "n_clusters" in metrics:
                    report_content += f"- **Number of Clusters:** {metrics.get('n_clusters', 0)}\n"
            
            # Add model parameters
            report_content += f"""
            ## Model Parameters
            """
            
            for param, value in st.session_state.get("hyperparams", {}).items():
                report_content += f"- **{param}:** {value}\n"
            
            # Allow download in selected formats
            report_downloads = []
            
            if "Text" in report_formats:
                report_downloads.append(
                    st.download_button(
                        "Download Text Report",
                        report_content,
                        file_name=f"model_report_{model_name}.txt",
                        mime="text/plain",
                        key="text_report"
                    )
                )
            
            if "HTML" in report_formats:
                # Convert report to HTML for download
                report_html = f"""
                <html>
                <head>
                    <title>Model Training Report - {model_name}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #4e54c8; }}
                        h2 {{ color: #4e54c8; margin-top: 30px; }}
                        .metric {{ margin: 5px 0; }}
                    </style>
                </head>
                <body>
                    <h1>Model Training Report</h1>
                    
                    <h2>Model Information</h2>
                    <div class="metric"><strong>Model Name:</strong> {model_name}</div>
                    <div class="metric"><strong>Model Type:</strong> {selected_model}</div>
                    <div class="metric"><strong>Problem Type:</strong> {problem_type.capitalize()}</div>
                    <div class="metric"><strong>Dataset:</strong> {st.session_state["selected_dataset"]["metadata"]["name"]}</div>
                    <div class="metric"><strong>Training Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
                    
                    <h2>Performance Metrics</h2>
                """
                
                # Add metrics based on problem type
                if problem_type == "classification":
                    report_html += f"""
                    <div class="metric"><strong>Accuracy:</strong> {metrics.get('accuracy', 0):.4f}</div>
                    <div class="metric"><strong>Precision:</strong> {metrics.get('precision', 0):.4f}</div>
                    <div class="metric"><strong>Recall:</strong> {metrics.get('recall', 0):.4f}</div>
                    <div class="metric"><strong>F1 Score:</strong> {metrics.get('f1', 0):.4f}</div>
                    """
                elif problem_type == "regression":
                    report_html += f"""
                    <div class="metric"><strong>R² Score:</strong> {metrics.get('r2', 0):.4f}</div>
                    <div class="metric"><strong>MAE:</strong> {metrics.get('mae', 0):.4f}</div>
                    <div class="metric"><strong>MSE:</strong> {metrics.get('mse', 0):.4f}</div>
                    <div class="metric"><strong>RMSE:</strong> {metrics.get('rmse', 0):.4f}</div>
                    """
                elif problem_type == "clustering":
                    if "silhouette" in metrics and metrics["silhouette"] is not None:
                        report_html += f'<div class="metric"><strong>Silhouette Score:</strong> {metrics.get("silhouette", 0):.4f}</div>\n'
                    if "inertia" in metrics:
                        report_html += f'<div class="metric"><strong>Inertia:</strong> {metrics.get("inertia", 0):.4f}</div>\n'
                    if "n_clusters" in metrics:
                        report_html += f'<div class="metric"><strong>Number of Clusters:</strong> {metrics.get("n_clusters", 0)}</div>\n'
                
                # Add model parameters
                report_html += "<h2>Model Parameters</h2>"
                
                for param, value in st.session_state.get("hyperparams", {}).items():
                    report_html += f'<div class="metric"><strong>{param}:</strong> {value}</div>\n'
                
                report_html += """
                </body>
                </html>
                """
                
                report_downloads.append(
                    st.download_button(
                        "Download HTML Report",
                        report_html,
                        file_name=f"model_report_{model_name}.html",
                        mime="text/html",
                        key="html_report"
                    )
                )
            
            if "JSON" in report_formats:
                # Create JSON report
                json_report = {
                    "model_info": {
                        "name": model_name,
                        "type": selected_model,
                        "problem_type": problem_type,
                        "dataset": st.session_state["selected_dataset"]["metadata"]["name"],
                        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    "metrics": metrics,
                    "parameters": st.session_state.get("hyperparams", {})
                }
                
                report_downloads.append(
                    st.download_button(
                        "Download JSON Report",
                        json.dumps(json_report, indent=2),
                        file_name=f"model_report_{model_name}.json",
                        mime="application/json",
                        key="json_report"
                    )
                )

# Additional sidebar options
with st.sidebar:
    st.markdown("## 🔧 Training Options")
    
    with st.expander("💡 Tips & Tricks"):
        st.markdown("""
        ### Training Tips
        
        1. **Data Splitting**: Use a larger training set (70-80%) for smaller datasets
        
        2. **Model Selection**: Start with simpler models (e.g., LinearRegression, LogisticRegression) as a baseline
        
        3. **Hyperparameters**: Try different hyperparameter values to optimize model performance
        
        4. **Metrics**: For classification, focus on precision/recall for imbalanced data; for regression, RMSE is often more interpretable than MSE
        """)