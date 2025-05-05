"""
ALPHA - End-to-End Machine Learning Platform
Main entry point for the application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from pathlib import Path

# Import utility modules
from utils.config import APP_CONFIG, get_available_datasets, get_available_models
from utils.ui import (
    set_page_config, welcome_header, create_card, display_footer,
    sidebar_navigation, gradient_text, create_feature_list, add_vertical_space
)

# Configure the page
set_page_config(title="Home", layout="wide")

# Display sidebar navigation
sidebar_navigation()

# Display user greeting in sidebar
st.sidebar.markdown("## 👋 Welcome")
st.sidebar.markdown(f"Welcome to ALPHA - your end-to-end ML platform built with Streamlit.")

# Help section in sidebar
with st.sidebar.expander("🤔 Need Help?"):
    st.markdown("""
    ALPHA provides an end-to-end Machine Learning workflow:
    
    1. Upload or generate datasets
    2. Explore and preprocess data
    3. Train and evaluate models
    4. Visualize results
    5. Deploy trained models
    
    Use the navigation menu to access different modules.
    """)

# Display platform status in sidebar
st.sidebar.markdown("## 📊 Platform Status")
datasets = get_available_datasets()
models = get_available_models()

col1, col2 = st.sidebar.columns(2)
col1.metric("Datasets", len(datasets))
col2.metric("Models", len(models))

# Main content
welcome_header()

# Quick actions section
st.markdown("## 🚀 Quick Actions")

# Create a grid of quick action cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #4e54c8, #7377de); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: white; margin-top: 0;">Upload Data</h3>
            <p style="color: white;">
                Upload your datasets to start your ML journey
            </p>
            <a href="/1_Data_Management" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; color: white;">
                    <span>Go to Data Management</span>
                    <span style="margin-left: 0.5rem;">→</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #ff6b6b, #ff9d9d); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: white; margin-top: 0;">Train Model</h3>
            <p style="color: white;">
                Train, evaluate, and save machine learning models
            </p>
            <a href="/2_Model_Training" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; color: white;">
                    <span>Go to Model Training</span>
                    <span style="margin-left: 0.5rem;">→</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #34eba8, #7af2c5); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: #2c3e50; margin-top: 0;">Deploy Model</h3>
            <p style="color: #2c3e50;">
                Deploy your trained models with one click
            </p>
            <a href="/5_Dashboard" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; color: #2c3e50;">
                    <span>Go to Dashboard</span>
                    <span style="margin-left: 0.5rem;">→</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

add_vertical_space(1)

# Second row of quick actions
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #A06CD5, #8E54E9); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: white; margin-top: 0;">🤗 Hugging Face</h3>
            <p style="color: white;">
                Explore and integrate state-of-the-art AI models
            </p>
            <a href="/11_Hugging_Face" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; color: white;">
                    <span>Go to Hugging Face Integration</span>
                    <span style="margin-left: 0.5rem;">→</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #20BDFF, #5AC8FA); 
                    padding: 1.5rem; 
                    border-radius: 0.8rem; 
                    height: 150px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
            <h3 style="color: white; margin-top: 0;">📊 Kaggle Datasets</h3>
            <p style="color: white;">
                Find and import datasets from Kaggle
            </p>
            <a href="/12_Kaggle_Datasets" target="_self" style="text-decoration: none;">
                <div style="display: flex; align-items: center; color: white;">
                    <span>Go to Kaggle Integration</span>
                    <span style="margin-left: 0.5rem;">→</span>
                </div>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

add_vertical_space(1)

# Platform overview
st.markdown("## 💡 Platform Features")

# Define all features
all_features = [
    {"icon": "📊", "name": "Natural Language Data Analysis"},
    {"icon": "🧠", "name": "Advanced ML Model Training"},
    {"icon": "📈", "name": "Interactive Visualizations"},
    {"icon": "🚀", "name": "One-Click Deployment Solutions"},
    {"icon": "🤖", "name": "RAG-Enhanced ML Assistant"},
    {"icon": "👁️", "name": "Computer Vision Capabilities"},
    {"icon": "🌐", "name": "ML Website Builder"},
    {"icon": "🔗", "name": "Hugging Face Integration"},
    {"icon": "🏆", "name": "Kaggle Dataset Access"},
    {"icon": "📱", "name": "Mobile-Friendly Interface"},
    {"icon": "🔒", "name": "Secure File-Based Storage"}
]

# Create a 3-column layout
cols = st.columns(3)

# Distribute features across columns without HTML rendering
for i, feature in enumerate(all_features):
    # Use modulo to determine which column to place the feature in
    col_index = i % 3
    
    # Add the feature to the appropriate column
    with cols[col_index]:
        st.container().markdown(f"**{feature['icon']} {feature['name']}**")

add_vertical_space(1)

# Recent Items Section
st.markdown("## 🕒 Recent Items")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Recent Datasets")
    if datasets:
        for dataset in datasets[:3]:
            st.markdown(
                f"""
                <div style="border-left: 4px solid #4e54c8; 
                            padding: 0.5rem 1rem; 
                            margin-bottom: 0.5rem;
                            background-color: rgba(78, 84, 200, 0.05);">
                    <div style="font-weight: bold;">{dataset.get('name', 'Unnamed Dataset')}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">{dataset.get('description', 'No description')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No datasets available. Start by uploading or generating a dataset.")

with col2:
    st.markdown("### Recent Models")
    if models:
        for model in models[:3]:
            perf = model.get('performance', {}).get('metrics', {})
            metric = next(iter(perf.items()), ('', ''))
            metric_text = f"{metric[0]}: {metric[1]:.4f}" if metric[1] else ""
            
            st.markdown(
                f"""
                <div style="border-left: 4px solid #ff6b6b; 
                            padding: 0.5rem 1rem; 
                            margin-bottom: 0.5rem;
                            background-color: rgba(255, 107, 107, 0.05);">
                    <div style="font-weight: bold;">{model.get('name', 'Unnamed Model')}</div>
                    <div style="font-size: 0.8rem; color: #6c757d;">
                        {model.get('type', 'Unknown type')} | {metric_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No models available. Train your first model in the Model Training module.")

add_vertical_space(1)

# Getting Started Guide
with st.expander("📚 Getting Started Guide", expanded=False):
    st.markdown("""
    ### Welcome to ALPHA!
    
    Follow these steps to get started with machine learning:
    
    1. **Prepare Your Data**: Upload your dataset or generate synthetic data in the Data Management module.
    
    2. **Explore & Preprocess**: Analyze your data, handle missing values, and prepare it for model training.
    
    3. **Train a Model**: Select an algorithm, tune hyperparameters, and train your model.
    
    4. **Evaluate Results**: Review model performance metrics and visualizations.
    
    5. **Deploy**: Generate deployment code or export your model for production use.
    
    Need more help? Use the ML Assistant for guidance throughout your journey.
    """)

# Footer
display_footer() 