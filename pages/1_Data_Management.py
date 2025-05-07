"""
ALPHA - End-to-End Machine Learning Platform
Data Management Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import io
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import utility modules
from utils.config import DATASETS_DIR, get_available_datasets
from utils.data import (
    save_uploaded_file, load_dataset, save_dataset, get_dataset_by_id,
    generate_synthetic_data, generate_dataset_summary, get_dataset_info
)
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box, 
    display_step_header, display_file_download_link, display_dataframe_with_download,
    create_tab_panels, display_json_viewer, add_vertical_space
)
from utils.imports import fix_dataframe_dtypes, logger, validate_dataframe_for_streamlit, is_package_available

# Configure the page
set_page_config(title="Data Management")

# Display sidebar navigation
sidebar_navigation()

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for data management. Please install pandas.')
    st.stop()
if not is_package_available('matplotlib'):
    st.warning('Matplotlib is not available. Some visualizations may not work.')
if not is_package_available('seaborn'):
    st.warning('Seaborn is not available. Some visualizations may not work.')

# Main content
page_header(
    title="Data Management",
    description="Upload, generate, explore, and manage your datasets",
    icon="📊"
)

# Display dataset tabs
dataset_tabs = create_tab_panels("Upload Dataset", "Generate Dataset", "Available Datasets")

# Tab 1: Upload Dataset
with dataset_tabs[0]:
    st.markdown("### Upload your own dataset")
    display_info_box(
        "Supported file formats: CSV, Excel, JSON, Parquet, and Pickle files",
        type="info"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["csv", "xlsx", "xls", "json", "parquet", "pkl", "pickle"]
    )
    
    # If a file is uploaded
    if uploaded_file is not None:
        try:
            # Show file details
            st.markdown(f"**File Details**:")
            col1, col2 = st.columns(2)
            col1.markdown(f"**Name:** {uploaded_file.name}")
            col2.markdown(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            # Save the file temporarily
            temp_file_path = save_uploaded_file(uploaded_file)
            
            # Load the data
            try:
                df = load_dataset(temp_file_path)
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
                    # Dataset info
                    with st.expander("Dataset Information", expanded=False):
                        st.markdown("##### Dataset Summary")
                        st.write(df.describe())
                        st.markdown("##### Column Types")
                        st.write(pd.DataFrame(df.dtypes, columns=["Type"]))
                        st.markdown("##### Missing Values")
                        missing_data = pd.DataFrame({
                            "Missing Values": df.isna().sum(),
                            "Percentage": round(df.isna().sum() / len(df) * 100, 2)
                        })
                        st.write(missing_data)
                    # Visualizations
                    with st.expander("Quick Visualizations", expanded=False):
                        viz_type = st.selectbox(
                            "Select visualization type",
                            ["Data Types", "Correlation Matrix", "Distribution Plots", "Missing Values"],
                        )
                        if viz_type == "Data Types":
                            if is_package_available('matplotlib'):
                                fig, ax = plt.subplots(figsize=(8, 6))
                                dtypes_counts = df.dtypes.value_counts()
                                ax.pie(dtypes_counts, labels=dtypes_counts.index, autopct="%1.1f%%", startangle=90)
                                ax.set_title("Data Types Distribution")
                                st.pyplot(fig)
                            else:
                                st.warning('Matplotlib is not available.')
                        elif viz_type == "Correlation Matrix":
                            numeric_df = df.select_dtypes(include=["int64", "float64"])
                            if len(numeric_df.columns) > 1 and is_package_available('seaborn'):
                                fig, ax = plt.subplots(figsize=(10, 8))
                                correlation = numeric_df.corr()
                                mask = np.triu(np.ones_like(correlation, dtype=bool))
                                sns.heatmap(correlation, mask=mask, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
                                ax.set_title("Feature Correlation")
                                st.pyplot(fig)
                            elif not is_package_available('seaborn'):
                                st.warning('Seaborn is not available.')
                            else:
                                st.info("Not enough numeric columns for correlation analysis")
                        elif viz_type == "Distribution Plots":
                            numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                            if numeric_cols and is_package_available('seaborn'):
                                selected_col = st.selectbox("Select column", numeric_cols)
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                                ax.set_title(f"Distribution of {selected_col}")
                                st.pyplot(fig)
                            elif not is_package_available('seaborn'):
                                st.warning('Seaborn is not available.')
                            else:
                                st.info("No numeric columns available for distribution plots")
                        elif viz_type == "Missing Values":
                            if is_package_available('seaborn'):
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
                                ax.set_title("Missing Values")
                                st.pyplot(fig)
                            else:
                                st.warning('Seaborn is not available.')
                
                # Save dataset form
                st.markdown("### Save Dataset")
                col1, col2 = st.columns(2)
                dataset_name = col1.text_input("Dataset Name", value=os.path.splitext(uploaded_file.name)[0])
                dataset_description = col2.text_area("Description (optional)")
                
                # Save button
                if st.button("Save Dataset"):
                    if dataset_name:
                        with st.spinner("Saving dataset..."):
                            # Save dataset with metadata
                            result = save_dataset(
                                df=df,
                                dataset_name=dataset_name,
                                description=dataset_description,
                                dataset_type="uploaded"
                            )
                            
                            st.success(f"Dataset '{dataset_name}' saved successfully!")
                            st.markdown(f"**Dataset ID:** `{result['id']}`")
                            st.balloons()
                    else:
                        st.error("Please provide a name for the dataset")
            
            except Exception as e:
                st.error(f"Error loading dataset: {str(e)}")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 2: Generate Dataset
with dataset_tabs[1]:
    st.markdown("### Generate Synthetic Dataset")
    display_info_box(
        "Create synthetic datasets for testing and development",
        type="info"
    )
    
    # Dataset generation form
    col1, col2 = st.columns(2)
    
    # Dataset type selection
    dataset_type = col1.selectbox(
        "Dataset Type",
        ["classification", "regression", "clustering"],
        help="Type of dataset to generate"
    )
    
    # Dataset size
    n_samples = col2.slider(
        "Number of Samples",
        min_value=50,
        max_value=10000,
        value=1000,
        step=50,
        help="Number of samples (rows) in the dataset"
    )
    
    # Number of features
    n_features = col1.slider(
        "Number of Features",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        help="Number of features (columns) in the dataset"
    )
    
    # Additional parameters based on dataset type
    if dataset_type == "classification":
        n_classes = col2.slider(
            "Number of Classes",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            help="Number of target classes"
        )
        
        n_informative = col1.slider(
            "Number of Informative Features",
            min_value=1,
            max_value=n_features,
            value=min(5, n_features),
            step=1,
            help="Number of features that are actually useful for prediction"
        )
        
        class_sep = col2.slider(
            "Class Separation",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Higher values make the classes more separable"
        )
    
    elif dataset_type == "regression":
        noise = col2.slider(
            "Noise",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05,
            help="Standard deviation of Gaussian noise added to the output"
        )
        
        n_informative = col1.slider(
            "Number of Informative Features",
            min_value=1,
            max_value=n_features,
            value=min(5, n_features),
            step=1,
            help="Number of features that are actually useful for prediction"
        )
    
    elif dataset_type == "clustering":
        n_clusters = col2.slider(
            "Number of Clusters",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Number of clusters to generate"
        )
        
        cluster_std = col1.slider(
            "Cluster Standard Deviation",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1,
            help="Standard deviation of clusters"
        )
    
    # Dataset name and description
    dataset_name = st.text_input("Dataset Name", value=f"synthetic_{dataset_type}")
    dataset_description = st.text_area(
        "Description",
        value=f"Synthetic {dataset_type} dataset with {n_samples} samples and {n_features} features"
    )
    
    # Generate button
    if st.button("Generate Dataset"):
        with st.spinner("Generating dataset..."):
            try:
                # Generate the dataset
                if dataset_type == "classification":
                    df = generate_synthetic_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        data_type=dataset_type,
                        n_classes=n_classes,
                        n_informative=n_informative,
                        class_sep=class_sep
                    )
                elif dataset_type == "regression":
                    df = generate_synthetic_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        data_type=dataset_type,
                        n_informative=n_informative,
                        # noise is handled inside generate_synthetic_data if needed
                    )
                elif dataset_type == "clustering":
                    df = generate_synthetic_data(
                        n_samples=n_samples,
                        n_features=n_features,
                        data_type=dataset_type,
                        n_classes=n_clusters,  # n_classes is used as centers in clustering
                        cluster_std=cluster_std
                    )
                else:
                    raise ValueError(f"Unsupported dataset type: {dataset_type}")
                
                # Fix DataFrame for display
                df = fix_dataframe_dtypes(df)
                
                st.success(f"Generated a {dataset_type} dataset with {df.shape[0]} rows and {df.shape[1]} columns")
                
                # Display data preview
                st.markdown("#### Generated Dataset Preview")
                st.dataframe(df.head(10))
                
                # Basic dataset info
                st.markdown("#### Dataset Information")
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Size (KB)", round(df.memory_usage(deep=True).sum() / 1024, 2))
                
                # Dataset statistics
                with st.expander("Dataset Statistics", expanded=False):
                    st.write(df.describe())
                
                # Visualizations
                with st.expander("Quick Visualizations", expanded=False):
                    if dataset_type == "classification" or dataset_type == "clustering":
                        # If 2D or 3D, show scatter plot
                        if n_features >= 2:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            scatter = ax.scatter(
                                df.iloc[:, 0],
                                df.iloc[:, 1],
                                c=df.iloc[:, -1],
                                cmap="viridis",
                                alpha=0.8,
                            )
                            ax.set_xlabel(df.columns[0])
                            ax.set_ylabel(df.columns[1])
                            ax.set_title(f"2D Visualization of {dataset_type.capitalize()} Dataset")
                            plt.colorbar(scatter, ax=ax, label="Target" if dataset_type == "classification" else "Cluster")
                            st.pyplot(fig)
                    
                    elif dataset_type == "regression":
                        # Regression plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.scatter(df.iloc[:, 0], df["target"], alpha=0.5)
                        ax.set_xlabel(df.columns[0])
                        ax.set_ylabel("Target")
                        ax.set_title("Target vs. Feature 1")
                        st.pyplot(fig)
                
                # Save the dataset
                if st.button("Save Generated Dataset"):
                    with st.spinner("Saving dataset..."):
                        result = save_dataset(
                            df=df,
                            dataset_name=dataset_name,
                            description=dataset_description,
                            dataset_type=f"synthetic_{dataset_type}"
                        )
                        
                        st.success(f"Dataset '{dataset_name}' saved successfully!")
                        st.markdown(f"**Dataset ID:** `{result['id']}`")
                        st.balloons()
            
            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")

# Tab 3: Available Datasets
with dataset_tabs[2]:
    st.markdown("### Available Datasets")
    
    # Refresh button
    if st.button("Refresh Dataset List"):
        st.rerun()
    
    # Get available datasets
    datasets = get_available_datasets()
    
    if datasets:
        # Create table of datasets
        dataset_df = pd.DataFrame(datasets)
        
        # If dataset_df has a 'created_at' column, sort by it in descending order
        if 'created_at' in dataset_df.columns:
            dataset_df = dataset_df.sort_values(by='created_at', ascending=False)
        
        # Display only relevant columns in the table
        if 'path' in dataset_df.columns:
            dataset_df = dataset_df.drop(columns=['path'])
        
        st.dataframe(dataset_df)
        
        # Dataset selection for details
        selected_dataset_name = st.selectbox(
            "Select a dataset to view details",
            options=[dataset["name"] for dataset in datasets],
            index=0
        )
        
        # Find the selected dataset
        selected_dataset = next(
            (dataset for dataset in datasets if dataset["name"] == selected_dataset_name),
            None
        )
        
        if selected_dataset:
            # Extract dataset ID from path
            dataset_path = Path(selected_dataset["path"])
            dataset_id = dataset_path.name
            
            # Load the dataset
            dataset_info = get_dataset_by_id(dataset_id)
            
            if dataset_info and dataset_info["data"] is not None:
                df = dataset_info["data"]
                metadata = dataset_info["metadata"]
                
                # Display dataset details
                st.markdown(f"### Dataset: {selected_dataset_name}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", df.shape[0])
                col2.metric("Columns", df.shape[1])
                col3.metric("Size (KB)", round(df.memory_usage(deep=True).sum() / 1024, 2))
                
                # Dataset details tabs
                detail_tabs = st.tabs(["Preview", "Stats", "Visualizations", "Metadata", "Export"])
                
                # Preview tab
                with detail_tabs[0]:
                    st.markdown("### Dataset Preview")
                    df = dataset_info["data"]
                    # Fix DataFrame for display
                    df = fix_dataframe_dtypes(df)
                    st.dataframe(df.head(10))
                    
                    with st.expander("Column Information"):
                        col_info = pd.DataFrame({
                            "Type": df.dtypes,
                            "Non-Null Count": df.count(),
                            "Null Count": df.isnull().sum(),
                            "Null Percentage": round(df.isnull().sum() / len(df) * 100, 2),
                            "Unique Values": [df[col].nunique() for col in df.columns]
                        })
                        st.dataframe(col_info)
                
                # Stats tab
                with detail_tabs[1]:
                    st.markdown("#### Descriptive Statistics")
                    st.write(df.describe(include="all").transpose())
                    
                    # For numeric columns, show additional stats
                    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
                    if len(numeric_cols) > 0:
                        st.markdown("#### Numeric Columns")
                        
                        selected_num_col = st.selectbox(
                            "Select a numeric column",
                            options=numeric_cols
                        )
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean", round(df[selected_num_col].mean(), 2))
                        col2.metric("Median", round(df[selected_num_col].median(), 2))
                        col3.metric("Min", round(df[selected_num_col].min(), 2))
                        col4.metric("Max", round(df[selected_num_col].max(), 2))
                    
                    # For categorical columns, show value counts
                    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
                    if len(cat_cols) > 0:
                        st.markdown("#### Categorical Columns")
                        
                        selected_cat_col = st.selectbox(
                            "Select a categorical column",
                            options=cat_cols
                        )
                        
                        st.write(df[selected_cat_col].value_counts())
                        
                        # Show percentage distribution
                        fig, ax = plt.subplots(figsize=(10, 6))
                        df[selected_cat_col].value_counts(normalize=True).plot(
                            kind="bar", ax=ax
                        )
                        ax.set_ylabel("Percentage")
                        ax.set_title(f"Distribution of {selected_cat_col}")
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
                        st.pyplot(fig)
                
                # Visualizations tab
                with detail_tabs[2]:
                    viz_type = st.selectbox(
                        "Select visualization type",
                        ["Data Types", "Correlation Matrix", "Distribution Plots", "Missing Values"],
                    )
                    
                    if viz_type == "Data Types":
                        # Data types pie chart
                        fig, ax = plt.subplots(figsize=(8, 6))
                        dtypes_counts = df.dtypes.value_counts()
                        ax.pie(dtypes_counts, labels=dtypes_counts.index, autopct="%1.1f%%", startangle=90)
                        ax.set_title("Data Types Distribution")
                        st.pyplot(fig)
                    
                    elif viz_type == "Correlation Matrix":
                        # Correlation heatmap
                        numeric_df = df.select_dtypes(include=["int64", "float64"])
                        if len(numeric_df.columns) > 1:
                            fig, ax = plt.subplots(figsize=(10, 8))
                            correlation = numeric_df.corr()
                            mask = np.triu(np.ones_like(correlation, dtype=bool))
                            sns.heatmap(correlation, mask=mask, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
                            ax.set_title("Feature Correlation")
                            st.pyplot(fig)
                        else:
                            st.info("Not enough numeric columns for correlation analysis")
                    
                    elif viz_type == "Distribution Plots":
                        # Select a column for distribution plot
                        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                        if numeric_cols:
                            selected_col = st.selectbox(
                                "Select column for distribution plot", 
                                numeric_cols
                            )
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                            ax.set_title(f"Distribution of {selected_col}")
                            st.pyplot(fig)
                            
                            # Show additional distribution stats
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Skewness", round(df[selected_col].skew(), 3))
                            col2.metric("Kurtosis", round(df[selected_col].kurtosis(), 3))
                            col3.metric("Std Dev", round(df[selected_col].std(), 3))
                            col4.metric("Variance", round(df[selected_col].var(), 3))
                        else:
                            st.info("No numeric columns available for distribution plots")
                    
                    elif viz_type == "Missing Values":
                        # Missing values heatmap
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis", ax=ax)
                        ax.set_title("Missing Values")
                        st.pyplot(fig)
                        
                        # Missing values percentage
                        missing_percentage = df.isnull().mean() * 100
                        missing_df = pd.DataFrame({
                            'Column': missing_percentage.index,
                            'Missing Percentage': missing_percentage.values
                        })
                        missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Missing Percentage', y='Column', data=missing_df, ax=ax)
                        ax.set_title("Missing Values Percentage by Column")
                        ax.set_xlim(0, 100)
                        st.pyplot(fig)
                
                # Metadata tab
                with detail_tabs[3]:
                    st.markdown("#### Dataset Metadata")
                    st.json(metadata)
                
                # Export tab
                with detail_tabs[4]:
                    st.markdown("#### Export Dataset")
                    export_format = st.selectbox(
                        "Select export format",
                        ["CSV", "Excel", "JSON", "Pickle"]
                    )
                    
                    # Button to generate the export
                    if st.button("Prepare Export"):
                        with st.spinner(f"Preparing {export_format} export..."):
                            if export_format == "CSV":
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_dataset_name}.csv",
                                    mime="text/csv"
                                )
                            
                            elif export_format == "Excel":
                                buffer = io.BytesIO()
                                df.to_excel(buffer, index=False)
                                buffer.seek(0)
                                st.download_button(
                                    label="Download Excel",
                                    data=buffer,
                                    file_name=f"{selected_dataset_name}.xlsx",
                                    mime="application/vnd.ms-excel"
                                )
                            
                            elif export_format == "JSON":
                                json_data = df.to_json(orient="records")
                                st.download_button(
                                    label="Download JSON",
                                    data=json_data,
                                    file_name=f"{selected_dataset_name}.json",
                                    mime="application/json"
                                )
                            
                            elif export_format == "Pickle":
                                buffer = io.BytesIO()
                                df.to_pickle(buffer)
                                buffer.seek(0)
                                st.download_button(
                                    label="Download Pickle",
                                    data=buffer,
                                    file_name=f"{selected_dataset_name}.pkl",
                                    mime="application/octet-stream"
                                )
            else:
                st.error("Error loading dataset data. The dataset might be corrupted.")
    else:
        st.info("No datasets available. Upload or generate a dataset first.")

# Additional sidebar filters and options
with st.sidebar:
    st.markdown("## 🔍 Module Actions")
    
    # Quick actions
    st.markdown("### Quick Actions")
    if st.button("Generate Sample Dataset"):
        with st.spinner("Generating sample dataset..."):
            # Generate a simple classification dataset
            df = generate_synthetic_data(
                n_samples=1000,
                n_features=5,
                data_type="classification"
            )
            
            # Save the dataset
            result = save_dataset(
                df=df,
                dataset_name="sample_dataset",
                description="Sample classification dataset for demonstration",
                dataset_type="synthetic_classification"
            )
            
            st.success("Sample dataset generated successfully!")
            
            # Switch to the Available Datasets tab
            dataset_tabs[2].selectbox(
                "Select a dataset to view details",
                options=["sample_dataset"],
                index=0
            )
    
    # Help section
    with st.expander("💡 Tips & Tricks"):
        st.markdown("""
        ### Data Management Tips
        
        1. **File Formats**: CSV is the most common format, but consider using Parquet for larger datasets
        
        2. **Data Quality**: Check for missing values and outliers before proceeding to model training
        
        3. **Synthetic Data**: Generate synthetic data to test your ML pipelines or when you need more training examples
        
        4. **Dataset Organization**: Use descriptive names and add detailed descriptions to your datasets
        """) 