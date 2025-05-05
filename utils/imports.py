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
            
        # Convert NumPy Float64DType to standard float64 for PyArrow compatibility
        for col in result_df.select_dtypes(include=['float']).columns:
            result_df[col] = result_df[col].astype('float64')
            
        # Convert NumPy Int64DType to standard int64
        for col in result_df.select_dtypes(include=['integer']).columns:
            result_df[col] = result_df[col].astype('int64')
            
        # Handle object columns with mixed types if needed
        for col in result_df.select_dtypes(include=['object']).columns:
            if pd.api.types.infer_dtype(result_df[col]) == 'mixed':
                # Convert to string to ensure compatibility
                result_df[col] = result_df[col].astype(str)
                
        return result_df
    except Exception as e:
        logger.warning(f"Error fixing DataFrame dtypes: {str(e)}")
        return df  # Return original DataFrame if conversion fails 