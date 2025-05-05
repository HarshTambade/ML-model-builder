"""
Utility module for handling optional imports in the ALPHA platform.
"""

import importlib
import logging
import pandas as pd
import numpy as np
import os
from utils.config import DEPENDENCY_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_package_available(package_name):
    """Check if a package is available/installed and enabled in configuration."""
    # Check if the package is disabled in configuration
    if package_name == "torch" and not DEPENDENCY_CONFIG["USE_PYTORCH"]:
        logger.info(f"Package {package_name} is disabled in configuration.")
        return False
    elif package_name == "tensorflow" and not DEPENDENCY_CONFIG["USE_TENSORFLOW"]:
        logger.info(f"Package {package_name} is disabled in configuration.")
        return False
    elif package_name in ["transformers", "huggingface_hub"] and not DEPENDENCY_CONFIG["USE_HUGGINGFACE"]:
        logger.info(f"Package {package_name} is disabled in configuration.")
        return False
    
    # Check if package is installed
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def get_optional_package(package_name, default=None):
    """Get an optional package, returning default if not available or disabled."""
    if not is_package_available(package_name):
        logger.warning(f"Optional package {package_name} not available. Some features may be limited.")
        return default
    
    try:
        return importlib.import_module(package_name)
    except Exception as e:
        logger.warning(f"Error importing package {package_name}: {str(e)}")
        return default

# Dictionary of optional package replacements
PACKAGE_ALTERNATIVES = {
    "torch": "sklearn",
    "tensorflow": "sklearn",
    "kaleido": None,
    "pyarrow": None
}

def get_package_or_alternative(package_name):
    """Get a package or its alternative if not available."""
    if is_package_available(package_name):
        return importlib.import_module(package_name)
    elif package_name in PACKAGE_ALTERNATIVES and PACKAGE_ALTERNATIVES[package_name]:
        alt_package = PACKAGE_ALTERNATIVES[package_name]
        logger.warning(f"Package {package_name} not found. Using {alt_package} instead.")
        return importlib.import_module(alt_package)
    else:
        logger.warning(f"Package {package_name} not found and no alternative available.")
        return None

# Helper functions for specific packages
def get_torch():
    """Get PyTorch if available and enabled, otherwise None."""
    if not DEPENDENCY_CONFIG["USE_PYTORCH"]:
        logger.info("PyTorch is disabled in configuration.")
        return None
    return get_optional_package("torch")

def get_tensorflow():
    """Get TensorFlow if available and enabled, otherwise None."""
    if not DEPENDENCY_CONFIG["USE_TENSORFLOW"]:
        logger.info("TensorFlow is disabled in configuration.")
        return None
    return get_optional_package("tensorflow")

def get_transformers():
    """Get Hugging Face Transformers if available and enabled, otherwise None."""
    if not DEPENDENCY_CONFIG["USE_HUGGINGFACE"]:
        logger.info("Hugging Face is disabled in configuration.")
        return None
    return get_optional_package("transformers")

def safe_import(module_name):
    """Decorator for safely importing a function from a module."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Check for disabled modules
            if module_name == "torch" and not DEPENDENCY_CONFIG["USE_PYTORCH"]:
                logger.info(f"Module {module_name} is disabled. Function {func.__name__} will not use it.")
                return None
            elif module_name == "tensorflow" and not DEPENDENCY_CONFIG["USE_TENSORFLOW"]:
                logger.info(f"Module {module_name} is disabled. Function {func.__name__} will not use it.")
                return None
            elif module_name in ["transformers", "huggingface_hub"] and not DEPENDENCY_CONFIG["USE_HUGGINGFACE"]:
                logger.info(f"Module {module_name} is disabled. Function {func.__name__} will not use it.")
                return None
                
            try:
                module = importlib.import_module(module_name)
                return func(module, *args, **kwargs)
            except ImportError:
                logger.warning(f"Module {module_name} not available. Function {func.__name__} will have limited functionality.")
                return None
        return wrapper
    return decorator

def fix_dataframe_dtypes(df):
    """Fix DataFrame dtypes to ensure compatibility with Arrow serialization.
    
    Addresses issues with numpy.dtypes.Float64DType and other types
    that cause PyArrow serialization errors in Streamlit.
    
    Args:
        df: Pandas DataFrame to fix
        
    Returns:
        DataFrame with compatible dtypes, or original input if not a DataFrame
    """
    try:
        # If it's not a DataFrame, return it as is
        if not isinstance(df, pd.DataFrame):
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Handle empty DataFrame
        if result_df.empty:
            return result_df
            
        # Convert NumPy float types to standard float64
        for col in result_df.select_dtypes(include=['float', 'float32', 'float64']).columns:
            result_df[col] = result_df[col].astype('float64')
            
        # Convert NumPy integer types to standard int64 where possible
        for col in result_df.select_dtypes(include=['integer', 'int32', 'int64']).columns:
            # Check for NaN values which can't be converted to int
            if result_df[col].isna().any():
                result_df[col] = result_df[col].astype('float64')
            else:
                result_df[col] = result_df[col].astype('int64')
        
        # Handle boolean columns
        for col in result_df.select_dtypes(include=['bool']).columns:
            result_df[col] = result_df[col].astype('bool')
            
        # Handle datetime columns
        for col in result_df.select_dtypes(include=['datetime']).columns:
            result_df[col] = pd.to_datetime(result_df[col])
            
        # Handle object/string columns with mixed types
        for col in result_df.select_dtypes(include=['object']).columns:
            # Try to infer better type
            inferred_type = pd.api.types.infer_dtype(result_df[col])
            
            if inferred_type == 'mixed':
                # Convert mixed types to string for safety
                result_df[col] = result_df[col].astype(str)
            elif inferred_type == 'mixed-integer':
                # Try to convert to float (which can represent integers with NaNs)
                try:
                    result_df[col] = result_df[col].astype('float64')
                except:
                    result_df[col] = result_df[col].astype(str)
            elif inferred_type == 'string':
                # Ensure strings are represented as 'object' dtype
                result_df[col] = result_df[col].astype('object')
                
        # Handle any complex types that might cause issues (like arrays in cells)
        for col in result_df.columns:
            if result_df[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
                # Convert complex types to strings
                result_df[col] = result_df[col].apply(lambda x: str(x) if isinstance(x, (list, dict, tuple)) else x)
                
        return result_df
        
    except Exception as e:
        logger.warning(f"Error fixing DataFrame dtypes: {str(e)}")
        # If we encounter errors, try to convert the entire DataFrame to simpler types
        try:
            # Last resort attempt: convert all columns to appropriate basic types
            for col in df.columns:
                try:
                    # Try to infer the type and convert appropriately
                    inferred_type = pd.api.types.infer_dtype(df[col])
                    if 'int' in inferred_type:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif 'float' in inferred_type:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    elif 'datetime' in inferred_type:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(str)
                except:
                    # If all else fails, convert to string
                    df[col] = df[col].astype(str)
            return df
        except:
            # If everything fails, return the original DataFrame
            return df 