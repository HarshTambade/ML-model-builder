"""
Data utility module for the ALPHA platform.
Contains functions for data loading, preprocessing, and manipulation.
"""

import os
import json
import time
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import pickle
from io import StringIO, BytesIO

from utils.config import DATASETS_DIR, TEMP_DIR, UPLOADS_DIR
from utils.imports import is_package_available, logger, fix_dataframe_dtypes

def get_dataset_list():
    """Get a list of all available datasets in the platform."""
    # Make sure the directory exists
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR, exist_ok=True)
        return []
    
    # List directories in the DATASETS_DIR, each representing a dataset
    dataset_dirs = [d for d in os.listdir(DATASETS_DIR) 
                  if os.path.isdir(os.path.join(DATASETS_DIR, d)) and 
                  os.path.exists(os.path.join(DATASETS_DIR, d, "metadata.json"))]
    
    return dataset_dirs

def get_dataset_info(dataset_id):
    """Get information about a dataset by its ID."""
    dataset_dir = os.path.join(DATASETS_DIR, dataset_id)
    
    if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
        return {"error": f"Dataset {dataset_id} not found"}
    
    # Try to load metadata
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {"error": f"Dataset {dataset_id} metadata not found"}
    
    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    except Exception as e:
        return {"error": f"Error loading dataset metadata: {str(e)}"}
    
    # Check for data files
    data_path = None
    for ext in [".csv", ".pkl", ".parquet", ".feather"]:
        potential_path = os.path.join(dataset_dir, f"data{ext}")
        if os.path.exists(potential_path):
            data_path = potential_path
            break
    
    if data_path is None:
        return {"error": f"Dataset {dataset_id} data file not found"}
    
    # Get basic stats without loading the full dataset
    stats = {}
    try:
        if data_path.endswith(".csv"):
            # Only read a few rows to get column information
            sample = pd.read_csv(data_path, nrows=5)
            stats["columns"] = list(sample.columns)
            # Get row count without loading entire dataset
            with open(data_path, 'r') as f:
                stats["rows"] = sum(1 for _ in f) - 1  # Subtract header
        elif data_path.endswith(".pkl"):
            # For pickle files, we need to load the entire dataset
            with open(data_path, 'rb') as f:
                sample = pickle.load(f)
            stats["rows"] = len(sample)
            stats["columns"] = list(sample.columns)
        # Add other formats as needed
    except Exception as e:
        stats["error_reading_stats"] = str(e)
    
    # Return the combined information
    return {
        "id": dataset_id,
        "name": metadata.get("name", dataset_id),
        "description": metadata.get("description", ""),
        "created_at": metadata.get("created_at", ""),
        "type": metadata.get("type", "unknown"),
        "file_path": data_path,
        "rows": metadata.get("size", stats.get("rows", 0)),
        "columns": metadata.get("columns", stats.get("columns", [])),
        "dtypes": metadata.get("dtypes", {}),
        "missing_values": metadata.get("missing_values", {})
    }

def load_dataset(dataset_name):
    """Load a dataset from storage."""
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return {"error": f"Dataset '{dataset_name}' not found. Please check the dataset name or upload it again."}
    # Find CSV file in dataset directory
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        return {"error": f"No CSV file found in dataset '{dataset_name}'. Please ensure the dataset was uploaded correctly."}
    try:
        # Load the first CSV file found
        df = pd.read_csv(csv_files[0])
    except pd.errors.ParserError as e:
        return {"error": f"CSV parsing error: {str(e)}. The file may be corrupted or not a valid CSV."}
    except FileNotFoundError:
        return {"error": f"CSV file not found in '{dataset_path}'."}
    except Exception as e:
        return {"error": f"Unexpected error loading dataset: {str(e)}"}
    # Fix DataFrame dtypes for ArrowInvalid serialization errors
    df = fix_dataframe_dtypes(df)
    return df

def save_uploaded_file(uploaded_file, directory=UPLOADS_DIR):
    """Save an uploaded file to the specified directory."""
    if uploaded_file is None:
        return None
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Generate a unique filename
    file_extension = os.path.splitext(uploaded_file.name)[1]
    unique_filename = f"{str(uuid.uuid4())}{file_extension}"
    file_path = os.path.join(directory, unique_filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def save_dataset(df, dataset_name, description="", dataset_type="tabular"):
    """Save a dataset with metadata."""
    # Create a directory for the dataset
    timestamp = int(time.time())
    dataset_id = f"{dataset_name.lower().replace(' ', '_')}_{timestamp}"
    dataset_dir = DATASETS_DIR / dataset_id
    os.makedirs(dataset_dir, exist_ok=True)
    # Fix DataFrame dtypes for ArrowInvalid serialization errors
    df = fix_dataframe_dtypes(df)
    try:
        # Save the dataset in multiple formats
        df.to_csv(dataset_dir / "data.csv", index=False)
        df.to_pickle(dataset_dir / "data.pkl")
    except PermissionError:
        return {"error": f"Permission denied when saving dataset to '{dataset_dir}'. Please check write permissions."}
    except Exception as e:
        return {"error": f"Error saving dataset: {str(e)}"}
    # Create a metadata file
    metadata = {
        "name": dataset_name,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": dataset_type,
        "size": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_values": df.isnull().sum().to_dict(),
        "id": dataset_id,
    }
    try:
        with open(dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        return {"error": f"Error saving dataset metadata: {str(e)}"}
    # Generate and save preview visualizations
    try:
        generate_dataset_visualizations(df, dataset_dir)
    except Exception as e:
        logger.warning(f"Could not generate dataset visualizations: {str(e)}")
    return {
        "id": dataset_id,
        "path": str(dataset_dir),
        "metadata": metadata
    }

def get_dataset_by_id(dataset_id):
    """Get a dataset by its ID."""
    dataset_dir = DATASETS_DIR / dataset_id
    
    if not dataset_dir.exists():
        return None
    
    # Load metadata
    with open(dataset_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load dataset
    try:
        if (dataset_dir / "data.pkl").exists():
            df = pd.read_pickle(dataset_dir / "data.pkl")
        elif (dataset_dir / "data.csv").exists():
            df = pd.read_csv(dataset_dir / "data.csv")
        else:
            df = None
    except:
        df = None
    
    return {
        "id": dataset_id,
        "path": str(dataset_dir),
        "metadata": metadata,
        "data": df
    }

def generate_dataset_visualizations(df, output_dir):
    """Generate basic visualizations for a dataset."""
    # Create visualizations directory
    viz_dir = Path(output_dir) / "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Data types distribution
    plt.figure(figsize=(10, 6))
    dtypes_counts = df.dtypes.value_counts()
    plt.pie(dtypes_counts, labels=dtypes_counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Data Types Distribution")
    plt.tight_layout()
    plt.savefig(viz_dir / "dtypes_distribution.png")
    plt.close()
    
    # 2. Missing values heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title("Missing Values")
    plt.tight_layout()
    plt.savefig(viz_dir / "missing_values.png")
    plt.close()
    
    # Limit to numerical columns for correlation
    num_df = df.select_dtypes(include=["int64", "float64"])
    if len(num_df.columns) > 1:
        # 3. Correlation heatmap
        plt.figure(figsize=(12, 10))
        correlation = num_df.corr()
        mask = np.triu(np.ones_like(correlation, dtype=bool))
        sns.heatmap(correlation, mask=mask, annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(viz_dir / "correlation.png")
        plt.close()
    
    # 4. Histograms for numerical features
    if len(num_df.columns) > 0:
        num_df.hist(figsize=(15, 10), bins=20)
        plt.suptitle("Numerical Features Distribution")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(viz_dir / "histograms.png")
        plt.close()

def preprocess_dataset(df, preprocessing_steps):
    """Apply preprocessing steps to a dataset."""
    processed_df = df.copy()
    
    # Keep track of transformers for later use
    transformers = {}
    
    # Apply each preprocessing step
    for step in preprocessing_steps:
        step_type = step.get("type")
        columns = step.get("columns", [])
        params = step.get("params", {})
        
        if step_type == "drop_columns":
            processed_df = processed_df.drop(columns=columns)
        
        elif step_type == "drop_missing":
            threshold = params.get("threshold", 0.5)
            # Drop columns with missing values above threshold
            if params.get("axis") == "columns":
                missing_pct = processed_df.isnull().mean()
                cols_to_drop = missing_pct[missing_pct > threshold].index
                processed_df = processed_df.drop(columns=cols_to_drop)
            # Drop rows with missing values above threshold
            else:
                processed_df = processed_df.dropna(thresh=int((1-threshold) * processed_df.shape[1]))
        
        elif step_type == "fill_missing":
            method = params.get("method", "mean")
            
            if method == "constant":
                value = params.get("value", 0)
                processed_df[columns] = processed_df[columns].fillna(value)
            
            elif method in ["mean", "median", "most_frequent"]:
                imputer = SimpleImputer(strategy=method)
                processed_df[columns] = imputer.fit_transform(processed_df[columns])
                transformers[f"imputer_{len(transformers)}"] = imputer
            
            elif method == "knn":
                n_neighbors = params.get("n_neighbors", 5)
                imputer = KNNImputer(n_neighbors=n_neighbors)
                processed_df[columns] = imputer.fit_transform(processed_df[columns])
                transformers[f"imputer_{len(transformers)}"] = imputer
        
        elif step_type == "scale":
            method = params.get("method", "standard")
            
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            processed_df[columns] = scaler.fit_transform(processed_df[columns])
            transformers[f"scaler_{len(transformers)}"] = scaler
        
        elif step_type == "encode":
            method = params.get("method", "onehot")
            
            if method == "label":
                for col in columns:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col])
                    transformers[f"label_encoder_{col}"] = le
            
            elif method == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded_data = encoder.fit_transform(processed_df[columns])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(columns))
                
                # Drop original columns and concat encoded ones
                processed_df = processed_df.drop(columns=columns).reset_index(drop=True)
                processed_df = pd.concat([processed_df, encoded_df], axis=1)
                
                transformers[f"onehot_encoder_{len(transformers)}"] = encoder
        
        elif step_type == "dim_reduction":
            method = params.get("method", "pca")
            n_components = params.get("n_components", min(5, len(columns)))
            
            if method == "pca":
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(processed_df[columns])
                pca_df = pd.DataFrame(
                    pca_result, 
                    columns=[f"PC{i+1}" for i in range(n_components)]
                )
                
                # Drop original columns and concat PCA ones
                processed_df = processed_df.drop(columns=columns).reset_index(drop=True)
                processed_df = pd.concat([processed_df, pca_df], axis=1)
                
                transformers[f"pca_{len(transformers)}"] = pca
    
    # Return processed dataframe and transformers dictionary
    return processed_df, transformers

def split_dataset(df, target_column, test_size=0.2, val_size=0.2, random_state=42):
    """Split a dataset into train, validation, and test sets."""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # First split: train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train and val from train_val
    # Adjust val_size to account for the test_size already removed
    adjusted_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, random_state=random_state
    )
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test
    }

def generate_synthetic_data(n_samples=100, n_features=5, data_type="classification", n_classes=2, n_informative=None, class_sep=1.0, cluster_std=1.0):
    """Generate synthetic dataset."""
    from sklearn.datasets import make_classification, make_regression, make_blobs
    
    # Default n_informative to half of n_features if not specified
    if n_informative is None:
        n_informative = max(1, n_features // 2)
    
    if data_type == "classification":
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(0, min(n_features - n_informative - 1, n_features // 4)),
            n_classes=n_classes,
            class_sep=class_sep,
            random_state=42
        )
        
        # Create feature names
        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        target_col = "target"
        
    elif data_type == "regression":
        X, y = make_regression(
            n_samples=n_samples, 
            n_features=n_features,
            n_informative=n_informative,
            noise=0.1,
            random_state=42
        )
        
        # Create feature names
        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        target_col = "target"
    
    elif data_type == "clustering":
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_classes,
            cluster_std=cluster_std,
            random_state=42
        )
        
        # Create feature names
        feature_cols = [f"feature_{i+1}" for i in range(n_features)]
        target_col = "cluster"
    
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_col] = y
    
    return df

def generate_dataset_summary(df):
    """Generate a summary of the dataset with key statistics."""
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_stats": {},
        "categorical_stats": {}
    }
    
    # Numeric columns
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            summary["categorical_stats"][col] = {
                "unique_values": df[col].nunique(),
                "value_counts": df[col].value_counts().to_dict()
            }
    
    return summary

def get_dataset_corr_matrix(df, method="pearson"):
    """Calculate the correlation matrix for a dataset."""
    # Filter only numeric columns
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    if len(numeric_df.columns) <= 1:
        return None
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    return corr_matrix.round(3)

def convert_df_to_various_formats(df, formats=None):
    """Convert DataFrame to various formats for download."""
    if formats is None:
        formats = ["csv", "json", "excel", "pickle"]
    
    # Fix DataFrame dtypes for ArrowInvalid serialization errors
    df = fix_dataframe_dtypes(df)
    
    results = {}
    
    if "csv" in formats:
        # CSV format
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        results["csv"] = buffer.getvalue()
    
    if "json" in formats:
        # JSON format
        buffer = StringIO()
        df.to_json(buffer, orient="records", date_format="iso")
        results["json"] = buffer.getvalue()
    
    if "excel" in formats:
        # Excel format
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        results["excel"] = buffer.getvalue()
    
    if "pickle" in formats:
        # Pickle format
        buffer = BytesIO()
        df.to_pickle(buffer)
        results["pickle"] = buffer.getvalue()
    
    return results

def load_dataset_from_path(file_path):
    """Load a dataset from a file path."""
    try:
        file_path = str(file_path)  # Convert Path objects to string
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
        else:
            return {"error": f"Unsupported file format: {file_path}"}
        
        # Fix DataFrame dtypes for ArrowInvalid serialization errors
        df = fix_dataframe_dtypes(df)
        
        return {
            "data": df,
            "file_path": file_path,
            "rows": len(df),
            "columns": list(df.columns),
            "shape": df.shape
        }
    except Exception as e:
        return {"error": f"Error loading dataset: {str(e)}"}

def get_dataset_info_from_df(df):
    """Get information about a DataFrame."""
    try:
        # Fix DataFrame dtypes for ArrowInvalid serialization errors
        df = fix_dataframe_dtypes(df)
        
        # Basic information
        info = {
            "rows": len(df),
            "columns": list(df.columns),
            "shape": df.shape,
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),  # in MB
        }
        
        # Data samples
        info["sample"] = df.head(5).to_dict(orient="records")
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            # Only calculate stats for numeric columns
            numeric_stats = df[numeric_cols].describe().to_dict()
            info["numeric_stats"] = numeric_stats
            info["numeric_columns"] = numeric_cols
        else:
            info["numeric_stats"] = {}
            info["numeric_columns"] = []
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        if categorical_cols:
            categorical_stats = {}
            for col in categorical_cols:
                if df[col].nunique() < 50:  # Only for columns with reasonable number of unique values
                    value_counts = df[col].value_counts().to_dict()
                    categorical_stats[col] = {
                        "unique_values": df[col].nunique(),
                        "top_values": {k: v for k, v in list(value_counts.items())[:10]}
                    }
            info["categorical_stats"] = categorical_stats
            info["categorical_columns"] = categorical_cols
        else:
            info["categorical_stats"] = {}
            info["categorical_columns"] = []
        
        return info
    
    except Exception as e:
        return {"error": f"Error analyzing dataset: {str(e)}"} 