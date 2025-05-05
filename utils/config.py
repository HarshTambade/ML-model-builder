"""
Configuration module for the ALPHA platform.
Contains global settings and paths used throughout the application.
"""

import os
import json
from pathlib import Path

# Configuration for optional dependencies that might cause issues
DEPENDENCY_CONFIG = {
    "USE_PYTORCH": os.environ.get("USE_PYTORCH", "false").lower() == "true",
    "USE_TENSORFLOW": os.environ.get("USE_TENSORFLOW", "false").lower() == "true",
    "USE_HUGGINGFACE": os.environ.get("USE_HUGGINGFACE", "true").lower() == "true",
}

# Base paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORAGE_DIR = ROOT_DIR / "storage"
DATASETS_DIR = STORAGE_DIR / "datasets"
MODELS_DIR = STORAGE_DIR / "models"
PROJECTS_DIR = STORAGE_DIR / "projects"
TEMP_DIR = STORAGE_DIR / "temp"
UPLOADS_DIR = STORAGE_DIR / "uploads"
DATA_DIR = DATASETS_DIR  # Alias for backward compatibility
SETTINGS_FILE = STORAGE_DIR / "settings.json"

# Ensure directories exist
for directory in [DATASETS_DIR, MODELS_DIR, PROJECTS_DIR, TEMP_DIR, UPLOADS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Application config
APP_CONFIG = {
    "name": "ALPHA",
    "version": "1.0.0",
    "description": "End-to-End Machine Learning Platform",
    "theme_color": "#4e54c8",
    "logo_emoji": "🧠",
}

# Default models config
DEFAULT_MODELS = {
    "classification": [
        "LogisticRegression", 
        "RandomForest", 
        "SVC", 
        "KNeighbors", 
        "DecisionTree", 
        "GradientBoosting", 
        "AdaBoost", 
        "XGBoost", 
        "LightGBM", 
        "ExtraTrees", 
        "NaiveBayes"
    ],
    "regression": [
        "LinearRegression", 
        "Ridge", 
        "Lasso", 
        "ElasticNet", 
        "RandomForestRegressor", 
        "GradientBoostingRegressor", 
        "SVR", 
        "KNeighborsRegressor", 
        "DecisionTreeRegressor", 
        "XGBoostRegressor", 
        "LightGBMRegressor"
    ],
    "clustering": [
        "KMeans", 
        "DBSCAN", 
        "AgglomerativeClustering", 
        "Birch", 
        "MeanShift", 
        "SpectralClustering", 
        "GaussianMixture"
    ],
    "nlp": [
        "BERT", 
        "DistilBERT", 
        "RoBERTa", 
        "GPT-2", 
        "T5", 
        "XLNet", 
        "BART", 
        "ALBERT", 
        "ELECTRA", 
        "LLaMa"
    ],
    "computer_vision": [
        "ResNet", 
        "VGG", 
        "EfficientNet", 
        "YOLO", 
        "SSD", 
        "Faster R-CNN", 
        "MobileNet", 
        "DenseNet", 
        "U-Net", 
        "ViT"
    ],
    "time_series": [
        "ARIMA", 
        "SARIMA", 
        "Prophet", 
        "LSTM", 
        "GRU", 
        "TCN", 
        "TransformerTimeSeries", 
        "StateSpaceModels", 
        "ExponentialSmoothing"
    ],
    "anomaly_detection": [
        "IsolationForest", 
        "LocalOutlierFactor", 
        "OneClassSVM", 
        "EllipticEnvelope", 
        "AutoEncoder"
    ],
    "recommender_systems": [
        "CollaborativeFiltering", 
        "MatrixFactorization", 
        "ContentBasedFiltering", 
        "HybridRecommender", 
        "DeepFM"
    ]
}

# Default preprocessing options
PREPROCESSING_OPTIONS = {
    "scaling": ["StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer"],
    "encoding": ["OneHotEncoder", "LabelEncoder", "OrdinalEncoder"],
    "feature_selection": ["PCA", "SelectKBest", "VarianceThreshold"],
    "missing_values": ["SimpleImputer", "KNNImputer", "IterativeImputer"],
}

# Hugging Face model categories
HUGGINGFACE_CATEGORIES = {
    "text-classification": "Text Classification",
    "token-classification": "Named Entity Recognition",
    "question-answering": "Question Answering",
    "summarization": "Text Summarization",
    "translation": "Translation",
    "text-generation": "Text Generation",
    "fill-mask": "Fill Mask",
    "image-classification": "Image Classification",
    "object-detection": "Object Detection",
    "image-segmentation": "Image Segmentation",
    "audio-classification": "Audio Classification",
    "automatic-speech-recognition": "Speech Recognition",
}

# File size limits (in MB)
FILE_SIZE_LIMITS = {
    "dataset_upload": 100,
    "image_upload": 10,
    "model_upload": 500,
}

# API Keys and external services configuration
# Get API keys from environment variables or use empty string as default
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")
KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "helllodigisir")
KAGGLE_API_KEY = os.environ.get("KAGGLE_API_KEY", "50ba11f56bdaa1ce0c7f3032c8629acf")

# Function to save user preferences
def save_user_preferences(preferences, user_id='default'):
    """Save user preferences to a JSON file."""
    pref_path = STORAGE_DIR / f"preferences_{user_id}.json"
    with open(pref_path, 'w') as f:
        json.dump(preferences, f)

# Function to load user preferences
def load_user_preferences(user_id='default'):
    """Load user preferences from a JSON file."""
    pref_path = STORAGE_DIR / f"preferences_{user_id}.json"
    if pref_path.exists():
        with open(pref_path, 'r') as f:
            return json.load(f)
    return {
        "theme": "light",
        "visualizations": {
            "default_chart_type": "bar",
            "color_scheme": "viridis",
            "display_grid": True,
        },
        "models": {
            "default_classification": "RandomForest",
            "default_regression": "LinearRegression",
        },
        "ui": {
            "sidebar_collapsed": False,
            "show_advanced_options": False,
        }
    }

# Function to get available dataset list
def get_available_datasets():
    """Get a list of available datasets in the datasets directory."""
    datasets = []
    for file_path in DATASETS_DIR.glob("*"):
        if file_path.is_dir():
            meta_path = file_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    datasets.append({
                        "name": metadata.get("name", file_path.name),
                        "path": str(file_path),
                        "description": metadata.get("description", ""),
                        "size": metadata.get("size", 0),
                        "created_at": metadata.get("created_at", ""),
                        "type": metadata.get("type", ""),
                    })
            else:
                datasets.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "description": "",
                    "size": 0,
                    "created_at": "",
                    "type": "",
                })
    return datasets

# Function to get available models
def get_available_models():
    """Get a list of available trained models in the models directory."""
    models = []
    for file_path in MODELS_DIR.glob("*"):
        if file_path.is_dir():
            meta_path = file_path / "metadata.json"
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    models.append({
                        "name": metadata.get("name", file_path.name),
                        "path": str(file_path),
                        "description": metadata.get("description", ""),
                        "type": metadata.get("type", ""),
                        "performance": metadata.get("performance", {}),
                        "created_at": metadata.get("created_at", ""),
                    })
    return models

# Function to load application settings
def load_settings():
    """Load application settings from a JSON file."""
    settings_path = STORAGE_DIR / "app_settings.json"
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            return json.load(f)
    
    # Default settings
    default_settings = {
        "general": {
            "theme": "light",
            "wide_mode": True,
            "show_tooltips": True,
            "notifications_enabled": True,
            "notification_level": "info"
        },
        "storage": {
            "data_directory": str(DATASETS_DIR),
            "models_directory": str(MODELS_DIR),
            "auto_cleanup": False,
            "cleanup_threshold_days": 30
        },
        "performance": {
            "enable_caching": True,
            "parallelism": 2,
            "logging_level": "INFO"
        },
        "developer": {
            "developer_mode": False,
            "allow_custom_code": False,
            "experimental_features": False
        },
        "github": {
            "connected": False,
            "username": "",
            "token_saved": False
        }
    }
    
    # Create default settings file
    with open(settings_path, 'w') as f:
        json.dump(default_settings, f, indent=2)
    
    return default_settings

# Function to save application settings
def save_settings(settings):
    """Save application settings to a JSON file."""
    settings_path = STORAGE_DIR / "app_settings.json"
    with open(settings_path, 'w') as f:
        json.dump(settings, f, indent=2)
    return True 