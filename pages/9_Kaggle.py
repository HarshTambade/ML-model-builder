"""
ALPHA - End-to-End Machine Learning Platform
Kaggle Integration Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space, display_code_block
)
from utils.kaggle import (
    check_kaggle_availability, initialize_kaggle_api, search_kaggle_datasets,
    download_kaggle_dataset, load_kaggle_dataset
)
from utils.data import (
    save_dataset, get_dataset_list, get_dataset_info
)
from utils.imports import is_package_available, logger

# Configure the page
set_page_config(title="Kaggle")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Kaggle Integration",
    description="Search, download, and use Kaggle datasets",
    icon="🏆"
)

# Check if Kaggle API is properly set up
kaggle_available = check_kaggle_availability()

if not kaggle_available:
    st.error("Kaggle API is not available. Please install the kaggle package using 'pip install kaggle'.")
    st.stop()

# Try initializing the API
api = initialize_kaggle_api()
if isinstance(api, dict) and "error" in api:
    st.error(f"Error initializing Kaggle API: {api['error']}")
    st.info("To set up Kaggle API credentials:")
    st.code("""
    1. Sign up for a Kaggle account at https://www.kaggle.com
    2. Go to your account settings (click on your profile picture → Account)
    3. Scroll down to 'API' section and click 'Create New API Token'
    4. Save the kaggle.json file in your home directory (~/.kaggle/kaggle.json)
    5. Update the configuration in config.py with your username and API key
    """)
    st.stop()

# Create tabs for different functionality
kaggle_tabs = create_tab_panels(
    "Dataset Search", "My Datasets", "Dataset Import"
)

# Tab 1: Dataset Search
with kaggle_tabs[0]:
    st.markdown("### 🔍 Search Kaggle Datasets")
    
    search_container = st.container()
    with search_container:
        # Search interface
        search_query = st.text_input("Search for datasets", placeholder="e.g., covid-19, titanic, mnist")
        
        col1, col2 = st.columns(2)
        with col1:
            search_limit = st.slider("Number of results", 5, 50, 20)
        with col2:
            search_sort = st.selectbox(
                "Sort by",
                options=["hottest", "votes", "updated", "relevance"]
            )
        
        search_button = st.button("Search Datasets")
        
        if search_button:
            with st.spinner("Searching Kaggle datasets..."):
                try:
                    # Perform search
                    results = search_kaggle_datasets(search_query, sort_by=search_sort, limit=search_limit)
                    
                    if isinstance(results, dict) and "error" in results:
                        st.error(f"Error: {results['error']}")
                    elif not results:
                        st.info("No datasets found matching your criteria.")
                    else:
                        st.session_state["search_results"] = results
                        st.success(f"Found {len(results)} datasets")
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
    
    # Display search results
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        
        st.markdown("### Search Results")
        
        # Display each dataset as a card
        for i, dataset in enumerate(results):
            # Get dataset name/title safely with a fallback
            dataset_title = dataset.get('title', dataset.get('name', f"Dataset {i+1}"))
            dataset_owner = dataset.get('ownerName', dataset.get('owner', 'Unknown'))
            
            with st.expander(f"{dataset_title} by {dataset_owner}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if "description" in dataset and dataset["description"]:
                        st.markdown(dataset["description"][:500] + "..." if len(dataset["description"]) > 500 else dataset["description"])
                    else:
                        st.markdown("No description available")
                    
                    st.markdown(f"**Last Updated:** {dataset.get('lastUpdated', 'Unknown')}")
                    st.markdown(f"**Downloads:** {dataset.get('downloadCount', 'Unknown')} | **Votes:** {dataset.get('voteCount', 'Unknown')}")
                    
                with col2:
                    # Dataset details and download options
                    # Check if 'ref' key exists, otherwise use other identifiers
                    dataset_ref = dataset.get('ref', dataset.get('id', dataset.get('name', f"dataset_{i}")))
                    st.markdown(f"**ID:** `{dataset_ref}`")
                    
                    # Actions
                    if st.button("Download & Import", key=f"download_{i}"):
                        with st.spinner(f"Downloading and importing dataset {dataset_ref}..."):
                            try:
                                # Download and load the dataset
                                result = load_kaggle_dataset(dataset_ref)
                                
                                if isinstance(result, dict) and "error" in result:
                                    st.error(f"Error: {result['error']}")
                                elif isinstance(result, dict) and "warning" in result:
                                    st.warning(f"Warning: {result['warning']}")
                                    # Show the sample data
                                    if "data" in result:
                                        df = result["data"]
                                        # Save the dataset
                                        save_name = f"kaggle_{dataset_ref.replace('/', '_')}"
                                        save_result = save_dataset(df, save_name, f"Kaggle dataset: {dataset['title']}")
                                        
                                        if isinstance(save_result, dict) and "error" in save_result:
                                            st.error(f"Error saving dataset: {save_result['error']}")
                                        else:
                                            st.success(f"Dataset imported and saved as '{save_name}'")
                                            st.dataframe(df.head(10))
                                else:
                                    # Successfully loaded the dataset
                                    df = result["data"]
                                    
                                    # Save the dataset
                                    save_name = f"kaggle_{dataset_ref.replace('/', '_')}"
                                    save_result = save_dataset(df, save_name, f"Kaggle dataset: {dataset['title']}")
                                    
                                    if isinstance(save_result, dict) and "error" in save_result:
                                        st.error(f"Error saving dataset: {save_result['error']}")
                                    else:
                                        st.success(f"Dataset imported and saved as '{save_name}'")
                                        st.dataframe(df.head(10))
                            except Exception as e:
                                st.error(f"Error downloading dataset: {str(e)}")
                    
                    st.markdown("[View on Kaggle](https://www.kaggle.com/datasets/{})".format(dataset_ref))

# Tab 2: My Datasets
with kaggle_tabs[1]:
    st.markdown("### 📊 Kaggle Datasets in Your Library")
    
    # Get all datasets that were imported from Kaggle
    datasets = get_dataset_list()
    kaggle_datasets = [d for d in datasets if d.startswith("kaggle_")]
    
    if not kaggle_datasets:
        st.info("You haven't imported any Kaggle datasets yet. Use the 'Dataset Search' tab to find and import datasets.")
    else:
        # Display imported Kaggle datasets
        for dataset_name in kaggle_datasets:
            # Get dataset info
            dataset_info = get_dataset_info(dataset_name)
            
            if dataset_info and "error" not in dataset_info:
                with st.expander(dataset_name):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {dataset_info.get('description', 'No description')}")
                        st.markdown(f"**Rows:** {dataset_info.get('rows', 'Unknown')} | **Columns:** {dataset_info.get('columns', 'Unknown')}")
                        st.markdown(f"**Created:** {dataset_info.get('created_at', 'Unknown')}")
                    
                    with col2:
                        # Actions
                        if st.button("Load Dataset", key=f"load_{dataset_name}"):
                            # Set the active dataset in session state
                            st.session_state["active_dataset"] = dataset_name
                            st.success(f"Dataset '{dataset_name}' loaded and set as active dataset")
                            st.info("Go to the 'Data Explorer' page to view and analyze this dataset")
                        
                        if st.button("Preview", key=f"preview_{dataset_name}"):
                            # Load and preview the dataset
                            try:
                                df = pd.read_csv(Path(dataset_info.get('file_path', '')))
                                st.dataframe(df.head(10))
                                
                                # Show column info
                                st.markdown("#### Column Information")
                                col_info = pd.DataFrame({
                                    "Type": df.dtypes,
                                    "Non-Null": df.count(),
                                    "Nulls": df.isnull().sum(),
                                    "Unique Values": [df[col].nunique() for col in df.columns]
                                })
                                st.dataframe(col_info)
                            except Exception as e:
                                st.error(f"Error loading dataset: {str(e)}")

# Tab 3: Dataset Import
with kaggle_tabs[2]:
    st.markdown("### 📥 Import Kaggle Dataset by Reference")
    
    # Import by reference
    st.markdown("""
    You can directly import a Kaggle dataset by providing its reference in the format: `owner/dataset-slug`.
    
    For example:
    - `uciml/iris` - The Iris dataset
    - `rtatman/chocolate-bar-ratings` - Chocolate Bar Ratings
    - `ronitf/heart-disease-uci` - Heart Disease UCI
    """)
    
    dataset_ref = st.text_input("Kaggle Dataset Reference", placeholder="e.g., uciml/iris")
    file_name = st.text_input("Specific File Name (Optional)", placeholder="Leave blank to auto-select")
    
    if file_name and not file_name.strip():
        file_name = None
    
    if st.button("Import Dataset"):
        if not dataset_ref or '/' not in dataset_ref:
            st.error("Please enter a valid dataset reference in the format 'owner/dataset-slug'")
        else:
            with st.spinner(f"Importing dataset {dataset_ref}..."):
                try:
                    # Import the dataset
                    result = load_kaggle_dataset(dataset_ref, file_name=file_name)
                    
                    if isinstance(result, dict) and "error" in result:
                        st.error(f"Error: {result['error']}")
                        
                        # If the error is about available files, show them
                        if "available_files" in result:
                            st.info("Available files in this dataset:")
                            for f in result["available_files"]:
                                st.markdown(f"- `{f}`")
                    else:
                        # Successfully loaded the dataset
                        df = result["data"]
                        
                        # Display dataset preview
                        st.success(f"Dataset loaded successfully: {result['shape'][0]} rows, {result['shape'][1]} columns")
                        st.dataframe(df.head(10))
                        
                        # Save option
                        save_name = st.text_input("Save Dataset As", value=f"kaggle_{dataset_ref.replace('/', '_')}")
                        save_description = st.text_area("Dataset Description", value=f"Kaggle dataset: {dataset_ref}")
                        
                        if st.button("Save Dataset"):
                            # Save the dataset
                            save_result = save_dataset(df, save_name, save_description)
                            
                            if isinstance(save_result, dict) and "error" in save_result:
                                st.error(f"Error saving dataset: {save_result['error']}")
                            else:
                                st.success(f"Dataset saved as '{save_name}'")
                                st.session_state["active_dataset"] = save_name
                                st.info("Go to the 'Data Explorer' page to view and analyze this dataset")
                except Exception as e:
                    st.error(f"Error importing dataset: {str(e)}")
    
    # Example datasets
    st.markdown("### Popular Datasets")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Iris Dataset"):
            st.session_state["dataset_ref_input"] = "uciml/iris"
            st.rerun()
    
    with col2:
        if st.button("Titanic Dataset"):
            st.session_state["dataset_ref_input"] = "titanic/titanic"
            st.rerun()
    
    with col3:
        if st.button("House Prices"):
            st.session_state["dataset_ref_input"] = "harlfoxem/housesalesprediction"
            st.rerun()

# Update text input if set from buttons
if "dataset_ref_input" in st.session_state:
    st.session_state["dataset_ref"] = st.session_state["dataset_ref_input"]
    del st.session_state["dataset_ref_input"] 