"""
Model utility module for the ALPHA platform.
Contains functions for model training, evaluation, and management.
"""

import os
import json
import time
import pickle
import uuid
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc,
    silhouette_score
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

from utils.config import MODELS_DIR, DATASETS_DIR
from utils.imports import is_package_available, get_optional_package, logger

# Check for optional packages availability
TENSORFLOW_AVAILABLE = is_package_available('tensorflow')
TORCH_AVAILABLE = is_package_available('torch')
TRANSFORMERS_AVAILABLE = is_package_available('transformers')
HUGGINGFACE_HUB_AVAILABLE = is_package_available('huggingface_hub')

# Import them only if they're available
if TENSORFLOW_AVAILABLE:
    tf = get_optional_package('tensorflow')

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    torch = get_optional_package('torch')
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    
    if HUGGINGFACE_HUB_AVAILABLE:
        from huggingface_hub import HfApi, HfFolder

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# For XGBoost, LightGBM, CatBoost we'll import them conditionally when needed
XGBClassifier = None
XGBRegressor = None
LGBMClassifier = None
LGBMRegressor = None
CatBoostClassifier = None
CatBoostRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    pass

def get_sklearn_model(model_name, model_type=None, **params):
    """Get a scikit-learn model instance by name with parameters."""
    # Basic scikit-learn models
    if model_name == "LogisticRegression":
        return LogisticRegression(**params)
    
    elif model_name == "RandomForest":
        return RandomForestClassifier(**params)
    
    elif model_name == "SVC":
        return SVC(**params)
    
    elif model_name == "KNeighbors":
        return KNeighborsClassifier(**params)
    
    elif model_name == "DecisionTree":
        return DecisionTreeClassifier(**params)
    
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(**params)
    
    elif model_name == "AdaBoost":
        return AdaBoostClassifier(**params)
    
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(**params)
    
    elif model_name == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**params)
    
    # Regression models
    elif model_name == "LinearRegression":
        return LinearRegression(**params)
    
    elif model_name == "Ridge":
        return Ridge(**params)
    
    elif model_name == "Lasso":
        return Lasso(**params)
    
    elif model_name == "ElasticNet":
        return ElasticNet(**params)
    
    elif model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    
    elif model_name == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**params)
    
    elif model_name == "SVR":
        return SVR(**params)
    
    elif model_name == "KNeighborsRegressor":
        return KNeighborsRegressor(**params)
    
    elif model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(**params)
    
    # Clustering models
    elif model_name == "KMeans":
        return KMeans(**params)
    
    elif model_name == "DBSCAN":
        return DBSCAN(**params)
    
    elif model_name == "AgglomerativeClustering":
        return AgglomerativeClustering(**params)
    
    elif model_name == "Birch":
        return Birch(**params)
    
    elif model_name == "MeanShift":
        return MeanShift(**params)
    
    elif model_name == "SpectralClustering":
        return SpectralClustering(**params)
    
    elif model_name == "GaussianMixture":
        return GaussianMixture(**params)
    
    # Anomaly detection models
    elif model_name == "IsolationForest":
        return IsolationForest(**params)
    
    elif model_name == "LocalOutlierFactor":
        return LocalOutlierFactor(**params)
    
    elif model_name == "OneClassSVM":
        return OneClassSVM(**params)
    
    elif model_name == "EllipticEnvelope":
        return EllipticEnvelope(**params)
    
    # Advanced models using optional dependencies
    elif model_name == "XGBoost" or model_name == "XGBoostRegressor":
        try:
            import xgboost as xgb
            model_class = xgb.XGBClassifier if model_name == "XGBoost" else xgb.XGBRegressor
            return model_class(**params)
        except ImportError:
            raise ImportError("XGBoost not installed. Install it with: pip install xgboost")
    
    elif model_name == "LightGBM" or model_name == "LightGBMRegressor":
        try:
            import lightgbm as lgb
            model_class = lgb.LGBMClassifier if model_name == "LightGBM" else lgb.LGBMRegressor
            return model_class(**params)
        except ImportError:
            raise ImportError("LightGBM not installed. Install it with: pip install lightgbm")
    
    # Time series models
    elif model_name in ["ARIMA", "SARIMA"]:
        try:
            import pmdarima as pm
            return pm.ARIMA(**params) if model_name == "ARIMA" else pm.SARIMAX(**params)
        except ImportError:
            raise ImportError("pmdarima not installed. Install it with: pip install pmdarima")
    
    elif model_name == "Prophet":
        try:
            from prophet import Prophet
            return Prophet(**params)
        except ImportError:
            raise ImportError("Prophet not installed. Install it with: pip install prophet")
    
    # If no model matches
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_sklearn_model(model, X, y=None):
    """
    Train a scikit-learn model and return the trained model.
    
    Args:
        model: A scikit-learn model instance
        X: Feature data
        y: Target data (not required for some unsupervised models)
        
    Returns:
        Trained model
    """
    try:
        # Check the type of model to determine the appropriate training method
        if isinstance(model, (RandomForestClassifier, LogisticRegression, SVC, DecisionTreeClassifier, 
                             KNeighborsClassifier, GradientBoostingClassifier, MLPClassifier)):
            # Classification models
            model.fit(X, y)
            
        elif XGBClassifier is not None and isinstance(model, XGBClassifier):
            model.fit(X, y)
            
        elif LGBMClassifier is not None and isinstance(model, LGBMClassifier):
            model.fit(X, y)
            
        elif CatBoostClassifier is not None and isinstance(model, CatBoostClassifier):
            model.fit(X, y)
            
        elif isinstance(model, (RandomForestRegressor, LinearRegression, SVR, DecisionTreeRegressor, 
                               KNeighborsRegressor, GradientBoostingRegressor, MLPRegressor, 
                               ElasticNet, Lasso, Ridge)):
            # Regression models
            model.fit(X, y)
            
        elif XGBRegressor is not None and isinstance(model, XGBRegressor):
            model.fit(X, y)
            
        elif LGBMRegressor is not None and isinstance(model, LGBMRegressor):
            model.fit(X, y)
            
        elif CatBoostRegressor is not None and isinstance(model, CatBoostRegressor):
            model.fit(X, y)
            
        elif isinstance(model, (KMeans, DBSCAN, AgglomerativeClustering)):
            # Clustering models
            if y is None:
                model.fit(X)
            else:
                # Some clustering algorithms can use y for semi-supervised approaches
                try:
                    model.fit(X, y)
                except TypeError:
                    # Fall back to unsupervised if y isn't supported
                    model.fit(X)
                    
        elif isinstance(model, (IsolationForest, LocalOutlierFactor, OneClassSVM)):
            # Anomaly detection models
            if y is None:
                model.fit(X)
            else:
                try:
                    model.fit(X, y)
                except TypeError:
                    model.fit(X)
        else:
            # Generic fallback
            try:
                if y is None:
                    model.fit(X)
                else:
                    model.fit(X, y)
            except Exception as e:
                logging.error(f"Failed to fit model with generic approach: {str(e)}")
                raise
            
        return model
    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        raise Exception(f"Failed to train model: {str(e)}")

def create_keras_model(input_shape, output_shape, model_type, **kwargs):
    """Create a Keras (TensorFlow) neural network model."""
    if not TENSORFLOW_AVAILABLE:
        logger.error("TensorFlow is not installed. Please install it to use Keras models.")
        raise ImportError("TensorFlow is not installed. Please install it to use Keras models.")
    
    model = tf.keras.Sequential()
    
    # Add input layer
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Add hidden layers
    hidden_layers = kwargs.get("hidden_layers", 2)
    neurons = kwargs.get("neurons", 64)
    activation = kwargs.get("activation", "relu")
    dropout_rate = kwargs.get("dropout_rate", 0.2)
    
    for i in range(hidden_layers):
        model.add(tf.keras.layers.Dense(neurons, activation=activation))
        if dropout_rate > 0:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Add output layer based on model type
    if model_type == "classification":
        if output_shape > 2:  # Multi-class
            model.add(tf.keras.layers.Dense(output_shape, activation="softmax"))
            loss = "categorical_crossentropy"
        else:  # Binary
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
    
    elif model_type == "regression":
        model.add(tf.keras.layers.Dense(output_shape, activation="linear"))
        loss = "mse"
    
    else:
        raise ValueError(f"Unsupported model type for Keras: {model_type}")
    
    # Compile model
    optimizer = kwargs.get("optimizer", "adam")
    metrics = kwargs.get("metrics", ["accuracy"] if model_type == "classification" else ["mae"])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    return model

def create_pytorch_model(input_size, hidden_sizes, output_size, model_type):
    """Create a PyTorch neural network model."""
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is not installed. Please install it to use PyTorch models.")
        raise ImportError("PyTorch is not installed. Please install it to use PyTorch models.")
    
    class NeuralNetwork(torch.nn.Module):
        def __init__(self, input_size, hidden_sizes, output_size, model_type):
            super(NeuralNetwork, self).__init__()
            
            layers = []
            
            # Input layer
            layers.append(torch.nn.Linear(input_size, hidden_sizes[0]))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
            
            # Hidden layers
            for i in range(len(hidden_sizes) - 1):
                layers.append(torch.nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(0.2))
            
            # Output layer
            layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))
            
            # Final activation based on model type
            if model_type == "classification":
                if output_size > 1:
                    layers.append(torch.nn.Softmax(dim=1))
                else:
                    layers.append(torch.nn.Sigmoid())
            
            self.model = torch.nn.Sequential(*layers)
        
        def forward(self, x):
            return self.model(x)
    
    # Create model instance
    model = NeuralNetwork(input_size, hidden_sizes, output_size, model_type)
    
    return model

def evaluate_classification_model(model, X_test, y_test):
    """Evaluate a classification model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For probability-based models, get class predictions
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        if y_prob.shape[1] > 2:  # Multi-class
            y_prob_positive = None  # Not applicable for multi-class
        else:  # Binary
            y_prob_positive = y_prob[:, 1]
    else:
        y_prob_positive = None
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Generate ROC curve and AUC for binary classification
    if y_prob_positive is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob_positive)
        metrics["auc"] = auc(fpr, tpr)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    else:
        roc_data = None
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "roc_data": roc_data
    }

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate a regression model and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }
    
    # Prepare residuals data
    residuals = y_test - y_pred
    residuals_data = {
        "residuals": residuals.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    
    return {
        "metrics": metrics,
        "residuals_data": residuals_data
    }

def evaluate_clustering_model(model, X):
    """Evaluate a clustering model and return metrics."""
    # Get cluster assignments
    if hasattr(model, "labels_"):
        labels = model.labels_
    else:
        labels = model.predict(X)
    
    # Number of clusters
    n_clusters = len(np.unique(labels))
    if -1 in labels:  # DBSCAN marks noise as -1
        n_clusters -= 1
    
    metrics = {
        "n_clusters": n_clusters,
    }
    
    # Calculate silhouette score if more than one cluster
    if n_clusters > 1 and -1 not in labels:
        try:
            silhouette = silhouette_score(X, labels)
            metrics["silhouette"] = silhouette
        except:
            metrics["silhouette"] = None
    
    # Cluster distribution
    cluster_counts = np.bincount(labels[labels >= 0])
    cluster_distribution = {
        f"cluster_{i}": count for i, count in enumerate(cluster_counts)
    }
    
    return {
        "metrics": metrics,
        "cluster_distribution": cluster_distribution,
        "labels": labels.tolist(),
    }

def save_model(model, model_name, model_type, metadata, metrics=None):
    """Save a trained model with metadata."""
    # Create a directory for the model
    timestamp = int(time.time())
    model_id = f"{model_name.lower().replace(' ', '_')}_{timestamp}"
    model_dir = MODELS_DIR / model_id
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model using pickle
    model_path = model_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Add additional metadata
    full_metadata = {
        "name": model_name,
        "type": model_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": model_id,
    }
    
    # Add user-provided metadata
    full_metadata.update(metadata)
    
    # Add performance metrics if provided
    if metrics:
        full_metadata["performance"] = metrics
    
    # Save metadata
    with open(model_dir / "metadata.json", "w") as f:
        json.dump(full_metadata, f, indent=2)
    
    return {
        "id": model_id,
        "path": str(model_dir),
        "metadata": full_metadata
    }

def load_model(model_id):
    """Load a model by its ID."""
    model_dir = MODELS_DIR / model_id
    
    if not model_dir.exists():
        return None
    
    # Load metadata
    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load model
    try:
        model_path = model_dir / "model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except:
        model = None
    
    return {
        "id": model_id,
        "path": str(model_dir),
        "metadata": metadata,
        "model": model
    }

def generate_model_visualizations(model, X, y=None, model_type="classification", output_dir=None):
    """Generate visualizations for a model."""
    if output_dir is None:
        return None
    
    viz_dir = Path(output_dir) / "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # For tree-based models
    if hasattr(model, "feature_importances_"):
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), importances[indices], align="center")
        plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(viz_dir / "feature_importances.png")
        plt.close()
    
    # For clustering models
    if model_type == "clustering" and hasattr(model, "labels_"):
        if X.shape[1] > 1:
            # 2D visualization of clusters (first 2 features)
            plt.figure(figsize=(10, 8))
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis")
            
            # If KMeans, add centroids
            if hasattr(model, "cluster_centers_"):
                plt.scatter(
                    model.cluster_centers_[:, 0],
                    model.cluster_centers_[:, 1],
                    s=300,
                    c="red",
                    marker="X",
                )
            
            plt.title("Cluster Visualization")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.tight_layout()
            plt.savefig(viz_dir / "clusters_2d.png")
            plt.close()
    
    # For classification models
    if model_type == "classification" and y is not None:
        # For classifiers with decision function or predict_proba
        if hasattr(model, "decision_function") or hasattr(model, "predict_proba"):
            # Create a 2D decision boundary plot if 2 features
            if X.shape[1] == 2:
                plt.figure(figsize=(10, 8))
                
                # Create a mesh grid
                h = 0.02  # step size in the mesh
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.arange(x_min, x_max, h),
                    np.arange(y_min, y_max, h)
                )
                
                # Make predictions on mesh grid
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                plt.contourf(xx, yy, Z, alpha=0.8, cmap="viridis")
                
                # Plot training points
                plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="viridis")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.title("Decision Boundary")
                plt.tight_layout()
                plt.savefig(viz_dir / "decision_boundary.png")
                plt.close()
    
    # For regression models
    if model_type == "regression" and y is not None:
        # Predicted vs Actual values
        y_pred = model.predict(X)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        plt.tight_layout()
        plt.savefig(viz_dir / "predicted_vs_actual.png")
        plt.close()
        
        # Residuals plot
        plt.figure(figsize=(10, 8))
        residuals = y - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors="red", linestyles="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.tight_layout()
        plt.savefig(viz_dir / "residuals.png")
        plt.close()
    
    return str(viz_dir)

def generate_deployment_code(model_info, file_type="python"):
    """Generate deployment code for a model."""
    model_name = model_info["metadata"]["name"]
    model_type = model_info["metadata"]["type"]
    
    if file_type == "python":
        code = f"""
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_data(data):
    \"\"\"
    Preprocess input data before making predictions.
    Adjust this function based on your specific preprocessing steps.
    \"\"\"
    # Convert input to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    # Apply preprocessing steps (example)
    # Example: data = data.fillna(0)
    
    return data

def predict(data):
    \"\"\"
    Make predictions with the trained model.
    
    Args:
        data: Input data for prediction
        
    Returns:
        Predictions from the model
    \"\"\"
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions

# Example usage
if __name__ == "__main__":
    # Example input data
    example_data = {{
        # Add example features here
    }}
    
    # Convert to DataFrame
    example_df = pd.DataFrame([example_data])
    
    # Get predictions
    result = predict(example_df)
    print(f"Prediction: {{result}}")
"""
    
    elif file_type == "fastapi":
        code = f"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Create FastAPI app
app = FastAPI(
    title="{model_name} API",
    description="API for making predictions with {model_name}",
    version="1.0.0"
)

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input data model
class InputData(BaseModel):
    # Define your input features here
    features: List[Dict[str, Any]]

# Define output data model
class PredictionResult(BaseModel):
    predictions: List[Any]
    
@app.post("/predict", response_model=PredictionResult)
def predict(data: InputData):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data.features)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Convert predictions to list
        predictions_list = predictions.tolist()
        
        return PredictionResult(predictions=predictions_list)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {{"message": f"Welcome to the {model_name} API", "model_type": f"{model_type}"}}

# Run with: uvicorn app:app --reload
"""
    
    elif file_type == "docker":
        code = f"""
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application files
COPY model.pkl .
COPY app.py .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    elif file_type == "requirements":
        code = f"""
fastapi>=0.100.0
uvicorn>=0.22.0
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.2.2
pydantic>=2.0.0
python-multipart>=0.0.6
"""
    
    else:
        code = "Unsupported file type"
    
    return code

def load_huggingface_model(model_name, task="text-classification"):
    """Load a model from Hugging Face."""
    if not TORCH_AVAILABLE:
        return {
            "error": "PyTorch and Transformers are not installed. Please install them to use Hugging Face models."
        }
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if task == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "task": task,
            "name": model_name
        }
    
    except Exception as e:
        return {
            "error": str(e)
        }

def huggingface_text_inference(model_info, text):
    """Run inference on text using a Hugging Face model."""
    if not TORCH_AVAILABLE:
        return {
            "error": "PyTorch and Transformers are not installed. Please install them to use Hugging Face models."
        }
    
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    task = model_info["task"]
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs based on task
    if task == "text-classification":
        # Get probabilities with softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get prediction class and confidence
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, predicted_class].item()
        
        # Get label if available
        if hasattr(model.config, "id2label"):
            label = model.config.id2label[predicted_class]
        else:
            label = f"CLASS_{predicted_class}"
        
        return {
            "label": label,
            "confidence": confidence,
            "all_probs": probs[0].tolist()
        }
    
    else:
        # For other tasks, return raw hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return {
            "embeddings": embeddings[0].tolist()
        }

def get_model_list():
    """Get a list of all available trained models."""
    # Make sure the directory exists
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR, exist_ok=True)
        return []
    
    # List directories in the MODELS_DIR, each representing a model
    model_dirs = [d for d in os.listdir(MODELS_DIR) 
                if os.path.isdir(os.path.join(MODELS_DIR, d)) and 
                os.path.exists(os.path.join(MODELS_DIR, d, "metadata.json"))]
    
    return model_dirs

def get_model_info(model_id):
    """Get information about a model by its ID."""
    model_dir = os.path.join(MODELS_DIR, model_id)
    
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        return {"error": f"Model {model_id} not found"}
    
    # Try to load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {"error": f"Model {model_id} metadata not found"}
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        return {"error": f"Error loading model metadata: {str(e)}"}
    
    # Check if model file exists
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        return {
            "id": model_id,
            "name": metadata.get("name", model_id),
            "type": metadata.get("type", "unknown"),
            "created_at": metadata.get("created_at", ""),
            "metadata": metadata,
            "error": "Model file not found"
        }
    
    # Return model info without loading the actual model (which could be large)
    return {
        "id": model_id,
        "name": metadata.get("name", model_id),
        "type": metadata.get("type", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "problem_type": metadata.get("problem_type", "unknown"),
        "created_at": metadata.get("created_at", ""),
        "performance": metadata.get("performance", {}),
        "parameters": metadata.get("parameters", {}),
        "metadata": metadata,
        "has_model_file": True
    } 