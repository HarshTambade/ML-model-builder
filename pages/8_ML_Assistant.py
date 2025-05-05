"""
ALPHA - End-to-End Machine Learning Platform
ML Assistant Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
import os

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)

# Import any active datasets
from utils.data import get_dataset_list, get_dataset_info, load_dataset

# Configure the page
set_page_config(title="ML Assistant")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="ML Assistant",
    description="Your AI-powered guide to machine learning",
    icon="🤖"
)

# Define rule-based response function for fallback
def get_rule_based_response(question, context=""):
    """Generate a rule-based response based on keywords in the question."""
    question_lower = question.lower()
    
    # Default responses based on question keywords
    responses = {
        "ml": """
        **Machine Learning (ML)** is a field of artificial intelligence that gives computers the ability to learn and improve from experience without being explicitly programmed.
        
        **Core Concepts**:
        
        1. **Learning from Data**: ML algorithms build mathematical models using training data to make predictions or decisions
        
        2. **Types of Machine Learning**:
           - **Supervised Learning**: Learning from labeled data (examples with correct answers)
           - **Unsupervised Learning**: Finding patterns in unlabeled data
           - **Reinforcement Learning**: Learning through trial and error with rewards/penalties
        
        3. **Key Components**:
           - **Data**: The foundation of any ML system
           - **Features**: The attributes or properties used for learning
           - **Models**: The mathematical representations learned from data
           - **Training**: The process of learning from data
           - **Inference**: Using trained models to make predictions
        
        4. **Common Applications**:
           - Image and speech recognition
           - Natural language processing
           - Recommendation systems
           - Fraud detection
           - Medical diagnosis
           - Autonomous vehicles
        
        5. **Popular ML Algorithms**:
           - Linear and Logistic Regression
           - Decision Trees and Random Forests
           - Support Vector Machines
           - Neural Networks and Deep Learning
           - K-means Clustering
           - Principal Component Analysis
        
        Machine learning has transformed industries by enabling systems to identify patterns, make decisions, and improve over time based on new data.
        """,
        
        "algorithm": """
        Choosing the right algorithm depends on:
        
        1. **Type of problem**: Classification, regression, clustering, etc.
        2. **Data characteristics**: Size, features, balance
        3. **Interpretability needs**: Some models are more explainable than others
        4. **Computational resources**: Some algorithms are more resource-intensive
        
        For **classification**:
        - Simple, interpretable: Decision Trees, Logistic Regression
        - High performance: Random Forest, XGBoost, Neural Networks
        
        For **regression**:
        - Linear relationships: Linear Regression
        - Complex patterns: Random Forest, Gradient Boosting, Neural Networks
        
        For **clustering**:
        - K-means: When you know the number of clusters
        - DBSCAN: For density-based clustering
        - Hierarchical: For hierarchical relationships
        """,
        
        "feature": """
        **Feature engineering** is the process of transforming raw data into features that better represent the underlying problem, resulting in improved model accuracy.
        
        Key techniques include:
        
        1. **Feature creation**: Creating new features from existing ones
           - Combining features (e.g., height/weight ratio)
           - Mathematical transformations (log, sqrt)
           - Date/time extraction (day of week, month, year)
        
        2. **Feature transformation**:
           - Scaling (MinMax, StandardScaler)
           - Encoding categorical variables (one-hot, label encoding)
           - Handling missing values
        
        3. **Feature selection**:
           - Removing irrelevant or redundant features
           - Using statistical methods (correlation, chi-square)
           - Model-based selection (feature importance)
        
        Good feature engineering often makes the difference between average and excellent model performance!
        """,
        
        "overfit": """
        **Overfitting** occurs when a model learns the training data too well - including its noise and outliers - resulting in poor generalization to new data.
        
        **Signs of overfitting**:
        - High accuracy on training data, low accuracy on test data
        - Complex model with many parameters
        - Model captures noise as if it were signal
        
        **Ways to prevent overfitting**:
        
        1. **Get more training data**: More data makes it harder to overfit
        
        2. **Use cross-validation**: Test on multiple subsets of data
        
        3. **Feature selection**: Remove irrelevant features
        
        4. **Regularization**: Add penalty for model complexity
           - L1 regularization (Lasso): Can zero out features
           - L2 regularization (Ridge): Shrinks parameters
        
        5. **Simpler models**: Choose less complex model architecture
        
        6. **Early stopping**: Stop training when validation error starts increasing
        
        7. **Ensemble methods**: Combine multiple models (e.g., Random Forest)
        
        8. **Pruning**: Remove parts of model that contribute to overfitting
        
        9. **Dropout**: Randomly ignore neurons during training (neural networks)
        
        The best approach depends on your specific situation and model type.
        """,
        
        "underfitting": """
        **Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. It performs poorly on both training and test data.
        
        **Signs of underfitting**:
        - Low accuracy on both training and test data
        - Model is too simple for the complexity of the data
        - High bias, low variance
        
        **Ways to address underfitting**:
        
        1. **Use more complex models**: Increase model capacity
        
        2. **Better feature engineering**: Create more informative features
        
        3. **Reduce regularization**: If using regularization, reduce its strength
        
        4. **Train longer**: Allow more training epochs/iterations
        
        5. **Add more features**: Include more relevant input features
        
        6. **Deep learning architectures**: For very complex patterns
        
        Balance is key - you want to avoid both underfitting and overfitting!
        """,
        
        "accuracy": """
        **Improving model accuracy** requires a systematic approach:
        
        1. **Data quality improvements**:
           - Clean data (remove/fix errors, handle missing values)
           - Collect more data
           - Balance imbalanced datasets (SMOTE, class weights)
        
        2. **Feature engineering**:
           - Create more informative features
           - Remove irrelevant features
           - Transform features appropriately
        
        3. **Model selection and tuning**:
           - Try different algorithms
           - Tune hyperparameters (Grid/Random search, Bayesian optimization)
           - Use ensemble methods (voting, stacking, boosting)
        
        4. **Advanced techniques**:
           - Cross-validation strategies
           - Feature selection methods
           - Learning rate schedules
           - Transfer learning for deep models
        
        5. **Evaluation strategy**:
           - Choose the right metrics for your problem
           - Use proper validation techniques
           - Consider different thresholds for classification
        
        Remember that the "best" model depends on your specific goals and constraints!
        """,
        
        "evaluate": """
        **Model evaluation** measures how well a model performs. The metrics depend on the problem type:
        
        **Classification metrics**:
        
        1. **Accuracy**: Percentage of correct predictions
           - Simple but misleading for imbalanced data
        
        2. **Precision**: True positives / (True positives + False positives)
           - Measures exactness, "of predicted positives, how many are actually positive"
        
        3. **Recall**: True positives / (True positives + False negatives)
           - Measures completeness, "of actual positives, how many were predicted"
        
        4. **F1 Score**: Harmonic mean of precision and recall
           - Balanced metric when classes are imbalanced
        
        5. **ROC-AUC**: Area under ROC curve
           - Measures discrimination ability across thresholds
        
        **Regression metrics**:
        
        1. **Mean Absolute Error (MAE)**: Average absolute difference
           - Robust to outliers, intuitive
        
        2. **Mean Squared Error (MSE)**: Average squared difference
           - Penalizes large errors more
        
        3. **Root Mean Squared Error (RMSE)**: Square root of MSE
           - Same units as target variable
        
        4. **R-squared (R²)**: Proportion of variance explained
           - Range: 0 to 1, higher is better
        
        **Validation techniques**:
        
        1. **Train-test split**: Simple division of data
        2. **Cross-validation**: Multiple splits for robustness
        3. **Time-series validation**: Respects temporal order
        4. **Leave-one-out**: For very small datasets
        """,
        
        "predict": """
        **Making predictions with a trained model** involves these steps:
        
        1. **Data preparation**:
           - Format new data just like the training data
           - Apply the same preprocessing (scaling, encoding, etc.)
           - Handle missing values consistently
        
        2. **Model loading**:
           - Load the saved model from disk or memory
           - Configure it for inference (vs. training)
        
        3. **Prediction**:
           - Call model.predict() on the prepared data
           - For probability estimates, use predict_proba() or similar
        
        4. **Post-processing**:
           - Apply threshold for binary classification if needed
           - Convert encoded predictions back to original form
           - Scale outputs back if target was scaled
        
        5. **Deployment considerations**:
           - Consider latency requirements
           - Batch vs. real-time predictions
           - Monitoring prediction performance
        
        In Python with scikit-learn, a basic prediction workflow is:
        ```python
        # Load model
        import joblib
        model = joblib.load('model.pkl')
        
        # Prepare data
        X_new = preprocess_data(new_data)
        
        # Make predictions
        predictions = model.predict(X_new)
        ```
        
        Always ensure your preprocessing pipeline for new data matches exactly what was used in training!
        """,
        
        "tune": """
        **Hyperparameter tuning** improves model performance by finding optimal settings:
        
        **Common hyperparameters by model type**:
        
        - **Random Forest**: n_estimators, max_depth, min_samples_split
        - **Gradient Boosting**: learning_rate, n_estimators, max_depth
        - **Neural Networks**: learning_rate, batch_size, number of layers, neurons per layer
        - **SVM**: C, kernel, gamma
        - **K-means**: n_clusters, initialization method
        
        **Tuning methods**:
        
        1. **Grid Search**:
           - Try all combinations of predefined parameter values
           - Exhaustive but computationally expensive
           ```python
           from sklearn.model_selection import GridSearchCV
           param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
           grid_search = GridSearchCV(estimator, param_grid, cv=5)
           ```
        
        2. **Random Search**:
           - Sample random combinations from parameter distributions
           - Often more efficient than grid search
           ```python
           from sklearn.model_selection import RandomizedSearchCV
           param_dist = {'n_estimators': range(100, 500), 'max_depth': range(5, 20)}
           random_search = RandomizedSearchCV(estimator, param_dist, n_iter=20, cv=5)
           ```
        
        3. **Bayesian Optimization**:
           - Builds a probabilistic model of the objective function
           - Efficient for expensive-to-evaluate models
           ```python
           # Using libraries like Optuna or scikit-optimize
           import optuna
           ```
        
        4. **Automated tools**:
           - Auto-sklearn, TPOT, Auto-Keras, etc.
           - Automate both model selection and hyperparameter tuning
        
        **Best practices**:
        - Use cross-validation to prevent overfitting during tuning
        - Start with broad search, then narrow down
        - Consider computational costs and budget time accordingly
        - Monitor for diminishing returns
        """,
        
        "metric": """
        **Performance metrics** help evaluate and compare machine learning models:
        
        **Classification metrics**:
        - **Accuracy**: Overall correctness (TP+TN)/(TP+TN+FP+FN)
        - **Precision**: Correctness of positive predictions TP/(TP+FP)
        - **Recall**: Completeness of positive predictions TP/(TP+FN)
        - **F1 Score**: Harmonic mean of precision and recall
        - **AUC-ROC**: Area under ROC curve, model's ability to distinguish classes
        - **Log Loss**: Penalizes confident incorrect predictions
        - **Confusion Matrix**: Table showing TP, TN, FP, FN counts
        
        **Regression metrics**:
        - **MAE**: Mean Absolute Error, average of absolute differences
        - **MSE**: Mean Squared Error, average of squared differences
        - **RMSE**: Root Mean Squared Error, square root of MSE
        - **R²**: Coefficient of determination, variance explained by model
        - **MAPE**: Mean Absolute Percentage Error, percentage-based error
        
        **Clustering metrics**:
        - **Silhouette Score**: Measures cluster cohesion and separation
        - **Davies-Bouldin Index**: Average similarity of clusters
        - **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
        
        **Ranking metrics**:
        - **NDCG**: Normalized Discounted Cumulative Gain
        - **MAP**: Mean Average Precision
        - **MRR**: Mean Reciprocal Rank
        
        Choose metrics based on your specific problem requirements!
        """,
        
        "dataset": """
        **Working with datasets** effectively is fundamental to machine learning success:
        
        **Data preparation steps**:
        
        1. **Data collection**:
           - Ensure data is relevant to your problem
           - Consider sample size and representativeness
        
        2. **Data cleaning**:
           - Handle missing values (imputation, removal)
           - Fix or remove corrupted values
           - Address outliers (remove, transform, or cap)
        
        3. **Exploratory Data Analysis (EDA)**:
           - Understand distributions and relationships
           - Identify patterns and anomalies
           - Generate hypotheses about features
        
        4. **Feature engineering**:
           - Create new features
           - Transform existing features
           - Select relevant features
        
        5. **Data splitting**:
           - Train/validation/test split (typically 70/15/15 or 80/10/10)
           - Consider stratification for imbalanced data
           - Ensure no data leakage between splits
        
        **Common issues and solutions**:
        
        - **Class imbalance**: Oversampling, undersampling, synthetic data, weighted losses
        - **Missing data**: Imputation, indicator features, models that handle NaN values
        - **High cardinality categoricals**: Encoding schemes, embedding layers
        - **Temporal data**: Careful validation, avoiding future leakage
        
        **File formats and tools**:
        
        - **CSV**: Simple, universal, but inefficient for large data
        - **Parquet/Arrow**: Columnar storage, efficient for analytics
        - **HDF5**: Good for hierarchical numerical data
        - **SQL**: For relational data with complex queries
        - **Pandas**: Most popular Python library for data manipulation
        - **Dask/Spark**: For distributed processing of large datasets
        
        Remember: "Garbage in, garbage out" - the quality of your dataset directly impacts model performance!
        """,
        
        "neural": """
        **Neural Networks** are machine learning models inspired by the human brain, composed of connected nodes (neurons) organized in layers.
        
        **Core components**:
        
        1. **Neurons**: Units that apply a non-linear function to weighted inputs
        2. **Layers**:
           - Input layer: Receives data
           - Hidden layers: Perform transformations
           - Output layer: Produces predictions
        3. **Activation functions**: ReLU, sigmoid, tanh, etc.
        4. **Weights and biases**: Parameters learned during training
        
        **How they work**:
        
        1. **Forward propagation**: Data flows through the network, with each layer transforming it
        2. **Loss calculation**: Output is compared to true values
        3. **Backpropagation**: Error is propagated backward to adjust weights
        4. **Optimization**: Weights updated using algorithms like gradient descent
        
        **Popular neural network types**:
        
        - **Feedforward (MLP)**: Basic architecture, fully connected layers
        - **Convolutional (CNN)**: Specialized for spatial data (images)
        - **Recurrent (RNN/LSTM/GRU)**: For sequential data (text, time series)
        - **Transformers**: Self-attention based networks for NLP and more
        - **Graph Neural Networks**: For data with graph structure
        
        **Popular frameworks**:
        
        - **TensorFlow/Keras**: Full-featured, production-ready
        - **PyTorch**: Dynamic computation graph, research-friendly
        - **JAX**: Functional approach with GPU/TPU optimization
        
        **Common architectures**:
        
        - **CNNs**: For images (ResNet, VGG, EfficientNet)
        - **RNNs/LSTMs**: For sequence data
        - **Transformers**: For NLP (BERT, GPT, T5)
        - **VAEs/GANs**: For generative modeling
        
        **Practical considerations**:
        
        - **Hardware requirements**: GPUs/TPUs accelerate training
        - **Data requirements**: Often needs large datasets
        - **Interpretability**: Often "black box"
        - **Deployment challenges**: Model size, inference speed
        
        **When to use deep learning**:
        
        - Complex problems (vision, NLP, speech)
        - Large datasets available
        - When traditional ML methods underperform
        - When feature engineering is difficult
        
        Deep learning has revolutionized AI but isn't always necessary - simpler models often work well for structured data!
        """
    }
    
    # Check for keywords in the question and return the relevant response
    for keyword, response in responses.items():
        if keyword in question_lower:
            # Add dataset context if available
            if context:
                return context + "\n\n" + response.strip()
            return response.strip()
    
    # Default response if no keyword matches
    return """
    I'd be happy to help with your machine learning question. To provide the most relevant information, 
    could you specify what aspect of machine learning you're interested in? 
    
    For example:
    - Choosing algorithms
    - Data preprocessing
    - Feature engineering
    - Model evaluation
    - Hyperparameter tuning
    - Deployment strategies
    
    Feel free to ask about specific techniques or concepts!
    """.strip()

# Create sidebar for context
with st.sidebar:
    st.markdown("### Assistant Context")
    
    # Get datasets for context
    datasets = get_dataset_list()
    if datasets:
        selected_dataset = st.selectbox("Select a dataset for context", ["None"] + datasets)
        if selected_dataset != "None":
            dataset_info = get_dataset_info(selected_dataset)
            if dataset_info and "error" not in dataset_info:
                st.success(f"Dataset '{selected_dataset}' loaded for context")
                # Load dataset info into session state for assistant context
                st.session_state["assistant_dataset"] = selected_dataset
                st.session_state["assistant_dataset_info"] = dataset_info
            else:
                st.error(f"Error loading dataset: {dataset_info.get('error', 'Unknown error')}")
    else:
        st.info("No datasets available. Import a dataset first.")
    
    # Show status of Hugging Face API
    st.markdown("### External Services")
    
    # Check Hugging Face availability
    try:
        from utils.huggingface import check_hf_availability, initialize_hf_api_key
        hf_status = check_hf_availability()
        
        if hf_status["status"] == "fully_available":
            st.success("✅ Hugging Face API is available and working")
        elif hf_status["status"] == "api_only":
            st.warning("⚠️ Hugging Face API is available, but transformers library is not installed")
            st.info("Install transformers for advanced ML Assistant features: pip install transformers")
        else:
            st.error("❌ Hugging Face API is not available or not working")
            
            # Show API key input field if no valid API key is found
            st.markdown("##### Enter Hugging Face API Key")
            temp_api_key = st.text_input("Enter a temporary Hugging Face API key:", type="password")
            
            if temp_api_key:
                # Store in session state
                st.session_state["temp_huggingface_api_key"] = temp_api_key
                
                # Test if the key works
                import os
                os.environ["HUGGINGFACE_API_KEY"] = temp_api_key
                
                # Recheck availability with new key
                updated_status = check_hf_availability()
                if updated_status["api_working"]:
                    st.success("✅ API key is valid and working!")
                else:
                    st.error("❌ API key verification failed. Please check your key.")
            
            # Instructions to get API key
            with st.expander("How to get an API key"):
                st.markdown("""
                1. Go to [huggingface.co](https://huggingface.co) and create an account
                2. Go to Settings > Access Tokens
                3. Create a new token with "Read" permission
                4. Copy and paste the token here
                """)
    
    except ImportError:
        st.error("❌ Hugging Face integration module not available")
    
    # Show information about rule-based fallback
    st.info("ML Assistant is using rule-based responses for reliable answers to common questions.")
    
    # Instructions to set up API key permanently
    with st.expander("Set up API key permanently"):
        st.markdown("""
        1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
        2. Generate an API token in your Account Settings
        3. Set the environment variable: `HUGGINGFACE_API_KEY=your_token_here`
        4. Or update the `utils/config.py` file to include your API key
        """)

# Create chat container
st.markdown("### ML Assistant Chat")
chat_container = st.container()
suggestion_container = st.container()

# Suggestion buttons for common questions
with suggestion_container:
    st.markdown("**Suggested questions:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("How do I choose the right algorithm?"):
            st.session_state["user_question"] = "How do I choose the right algorithm for my machine learning task?"
    
    with col2:
        if st.button("Explain feature engineering"):
            st.session_state["user_question"] = "Can you explain what feature engineering is and why it's important?"
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("How can I improve model accuracy?"):
            st.session_state["user_question"] = "How can I improve the accuracy of my machine learning model?"
    
    with col4:
        if st.button("What is overfitting?"):
            st.session_state["user_question"] = "What is overfitting and how can I prevent it?"
            
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button("How to tune hyperparameters?"):
            st.session_state["user_question"] = "What are the best methods for hyperparameter tuning?"
    
    with col6:
        if st.button("Explain neural networks"):
            st.session_state["user_question"] = "Can you explain how neural networks work?"

# Placeholder for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your ML Assistant. Ask me anything about machine learning, or select a dataset from the sidebar to get contextual help."}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask a question about machine learning...")

# Set user_question from session state if it exists (from suggestions)
if "user_question" in st.session_state:
    user_question = st.session_state["user_question"]
    # Clear it after use to avoid repetition
    del st.session_state["user_question"]
    # Force rerun to show the question in the chat box
    st.rerun()

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)
    
    # Generate an answer
    with st.spinner("Thinking..."):
        # Get dataset context if available
        dataset_context = ""
        if "assistant_dataset" in st.session_state:
            dataset_name = st.session_state["assistant_dataset"]
            dataset_info = st.session_state.get("assistant_dataset_info", {})
            dataset_context = f"Dataset context: {dataset_name} has {dataset_info.get('rows', 'unknown')} rows and {dataset_info.get('columns', 'unknown')} columns."
        
        # Use rule-based responses
        assistant_response = get_rule_based_response(user_question, dataset_context)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # Display assistant message
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

# Footer with usage info
st.markdown("---")
st.markdown("""
**Tips for using the ML Assistant:**
* Ask specific questions about machine learning concepts
* Select a dataset from the sidebar for contextual help
* Use the suggested questions or type your own
* Focus on specific keywords like "algorithm", "feature", "overfit", "accuracy", etc.
* Browse suggested topics for ideas on what to ask
""")

# Features panel
with st.expander("ML Assistant Features", expanded=False):
    st.markdown("""
    ### Current Features
    
    The ML Assistant can help you with:
    
    * **ML Knowledge**: Get answers to your machine learning questions
    * **Dataset Context**: Get contextual help based on your selected dataset
    * **Rule-Based Responses**: Get reliable predefined responses for common questions
    
    ### ML Topics Covered
    
    * Algorithm selection
    * Feature engineering
    * Model evaluation
    * Hyperparameter tuning
    * Overfitting and underfitting
    * Neural networks and deep learning
    * Dataset preparation
    * Performance metrics
    * And many more!
    """)

# Development status
st.markdown("---")
st.markdown("**Module Status:** Active - Rule-based ML Assistant") 