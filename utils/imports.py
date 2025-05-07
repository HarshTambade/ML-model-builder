"""
Utility module for handling optional imports in the ALPHA platform.
"""

import importlib
import logging
import pandas as pd
import numpy as np
import os
import sys
from utils.config import DEPENDENCY_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def suppress_torch_warnings():
    """
    Suppress torch-related warnings from Streamlit's file watcher.
    This function creates monkey patches to prevent errors related to PyTorch.
    """
    try:
        # Completely disable torch-related modules in streamlit file watching
        import sys
        import types
        
        # Step 1: Create a better mock for torch.classes that implements __path__
        class MockPath:
            _path = []
            
        class MockTorchClasses:
            __path__ = MockPath()
            
            def __getattr__(self, name):
                return self
                
        # Step 2: Monkey patch the problematic function in Streamlit's file watcher
        try:
            import streamlit.watcher.local_sources_watcher as watcher
            
            # Save the original function
            original_extract_paths = watcher.extract_paths
            
            # Create a safer version that skips torch-related modules
            def safe_extract_paths(module):
                module_name = getattr(module, "__name__", "")
                if "torch" in module_name:
                    return []
                try:
                    return original_extract_paths(module)
                except Exception:
                    return []
            
            # Apply the monkey patch
            watcher.extract_paths = safe_extract_paths
            logger.info("Streamlit file watcher patched to ignore torch modules")
        except ImportError:
            # If we can't import watcher, use the mock approach
            if "torch" in sys.modules:
                sys.modules["torch.classes"] = MockTorchClasses()
                if hasattr(sys.modules["torch"], "classes"):
                    sys.modules["torch"].classes = MockTorchClasses()
                logger.info("Mock torch.classes module installed")
        
        # Step 3: Filter torch-related warnings from logs
        st_logger = logging.getLogger('streamlit')
        
        # Create a filter to remove torch-related warnings
        class TorchWarningFilter(logging.Filter):
            def filter(self, record):
                # Return False to suppress log messages containing these strings
                problematic_patterns = [
                    'torch.classes', 
                    'Tried to instantiate class', 
                    'RuntimeError: no running event loop',
                    'Could not convert dtype',
                    'ArrowInvalid',
                    'Arrow'
                ]
                message = str(record.getMessage())
                return not any(x in message for x in problematic_patterns)
                
        # Add the filter to all relevant loggers
        st_logger.addFilter(TorchWarningFilter())
        logging.getLogger().addFilter(TorchWarningFilter())
        
        # Set pandas display options to prevent truncation that might cause serialization issues
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        
        logger.info("Torch warning suppression activated")
    except Exception as e:
        logger.warning(f"Failed to suppress torch warnings: {str(e)}")

def torch_is_compatible():
    """Check if PyTorch is installed AND compatible with the current environment."""
    if not DEPENDENCY_CONFIG["USE_PYTORCH"]:
        return False
        
    try:
        # Try importing torch
        import torch
        
        # Try a simple operation to check compatibility
        try:
            x = torch.zeros(1)
            return True
        except Exception as e:
            logger.warning(f"PyTorch is installed but not compatible: {str(e)}")
            return False
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"PyTorch import raised an unexpected error: {str(e)}")
        return False

def is_package_available(package_name):
    """Check if a package is available/installed and enabled in configuration."""
    # Special case for PyTorch - check compatibility too
    if package_name == "torch":
        return torch_is_compatible()
        
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

def patch_streamlit_dataframe_display():
    """
    Patch Streamlit's DataFrame display functionality to handle serialization errors.
    This adds a wrapper around Streamlit's dataframe display methods.
    """
    try:
        import streamlit as st
        
        # Store original functions
        original_dataframe = st.dataframe
        original_table = st.table
        original_write = st.write
        
        # Create patched versions that apply fixes before display
        def patched_dataframe(data, *args, **kwargs):
            if isinstance(data, pd.DataFrame):
                is_valid, msg, problematic = validate_dataframe_for_streamlit(data)
                if not is_valid:
                    st.error(f"Cannot display DataFrame: {msg}")
                    return None
                # Apply fixes to make DataFrame compatible with Arrow
                fixed_data = fix_dataframe_dtypes(data)
                return original_dataframe(fixed_data, *args, **kwargs)
            return original_dataframe(data, *args, **kwargs)
        
        def patched_table(data, *args, **kwargs):
            if isinstance(data, pd.DataFrame):
                is_valid, msg, problematic = validate_dataframe_for_streamlit(data)
                if not is_valid:
                    st.error(f"Cannot display DataFrame: {msg}")
                    return None
                fixed_data = fix_dataframe_dtypes(data)
                return original_table(fixed_data, *args, **kwargs)
            return original_table(data, *args, **kwargs)
        
        def patched_write(*args, **kwargs):
            if len(args) > 0 and isinstance(args[0], pd.DataFrame):
                is_valid, msg, problematic = validate_dataframe_for_streamlit(args[0])
                if not is_valid:
                    st.error(f"Cannot display DataFrame: {msg}")
                    return None
                fixed_data = fix_dataframe_dtypes(args[0])
                return original_write(fixed_data, *args[1:], **kwargs)
            return original_write(*args, **kwargs)
        
        # Replace the original functions with patched versions
        st.dataframe = patched_dataframe
        st.table = patched_table
        st.write = patched_write
        
        logger.info("Streamlit DataFrame display patched for better compatibility")
    except Exception as e:
        logger.warning(f"Failed to patch Streamlit DataFrame display: {str(e)}")

def apply_torch_monkey_patch():
    """
    Apply a comprehensive monkey patch to disable PyTorch-related functionality
    and prevent errors when PyTorch is installed but not working correctly.
    This is a more extreme solution than suppress_torch_warnings.
    """
    import sys
    import types
    
    # Only apply if PyTorch is disabled in configuration
    if DEPENDENCY_CONFIG.get("USE_PYTORCH", False):
        logger.info("PyTorch is enabled, not applying monkey patch")
        return
        
    logger.info("Applying PyTorch monkey patch to prevent errors")
    
    # Create a complete mock torch module
    class MockTorch(types.ModuleType):
        def __init__(self):
            super().__init__("torch")
            self._C = types.SimpleNamespace()
            self._C._get_custom_class_python_wrapper = lambda *args, **kwargs: None
            self.classes = types.SimpleNamespace()
            self.classes.__path__ = types.SimpleNamespace(_path=[])
        
        def __getattr__(self, name):
            # Return empty modules/functions for any attributes
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(f"module 'torch' has no attribute '{name}'")
            
            # Create a new namespace for submodules
            value = types.SimpleNamespace()
            setattr(self, name, value)
            return value
            
    # Replace torch in sys.modules if it exists
    if "torch" in sys.modules:
        # Save any critical functionality if needed
        old_torch = sys.modules["torch"]
        
        # Install mock
        sys.modules["torch"] = MockTorch()
        logger.info("PyTorch module completely replaced with mock")
    
    # Monkey patch streamlit to avoid scanning torch modules
    try:
        # Try to patch streamlit's file watcher directly
        import streamlit.watcher.local_sources_watcher as watcher
        
        # Save original functions
        if not hasattr(watcher, "_original_get_module_paths"):
            watcher._original_get_module_paths = watcher.get_module_paths
        
        # Create safe versions that skip torch
        def safe_get_module_paths(filenames):
            def is_not_torch_module(module_name):
                return "torch" not in module_name
                
            # Filter out torch-related modules before calling original
            safe_modules = filter(is_not_torch_module, watcher._original_get_module_paths(filenames))
            return list(safe_modules)
            
        # Apply patches
        watcher.get_module_paths = safe_get_module_paths
        logger.info("Streamlit file watcher patched to completely ignore torch modules")
    except (ImportError, AttributeError) as e:
        logger.warning(f"Could not patch Streamlit watcher: {str(e)}")
    
    # Suppress Arrow serialization errors
    try:
        if "pyarrow" in sys.modules:
            # Monkey patch pandas to_arrow to handle problematic conversions
            original_to_arrow = pd.DataFrame.to_arrow
            
            def safe_to_arrow(self, *args, **kwargs):
                try:
                    # Try original conversion
                    return original_to_arrow(self, *args, **kwargs)
                except Exception as e:
                    # If it fails, fix dtypes and try again
                    fixed_df = fix_dataframe_dtypes(self)
                    return original_to_arrow(fixed_df, *args, **kwargs)
                    
            pd.DataFrame.to_arrow = safe_to_arrow
            logger.info("Fixed pandas to_arrow conversion for better Arrow compatibility")
    except Exception as e:
        logger.warning(f"Could not patch pandas arrow conversion: {str(e)}")
        
    return True 

def validate_dataframe_for_streamlit(df):
    """
    Check a DataFrame for columns with unsupported types for Streamlit/Arrow serialization.
    Returns a tuple (is_valid, message, problematic_columns)
    """
    if not isinstance(df, pd.DataFrame):
        return True, "Input is not a DataFrame.", []
    problematic = []
    for col in df.columns:
        dtype = df[col].dtype
        inferred = pd.api.types.infer_dtype(df[col], skipna=True)
        if inferred in ["mixed", "mixed-integer", "mixed-integer-float", "complex", "mixed-integer-float", "mixed-integer-float"]:
            problematic.append((col, str(dtype), inferred))
        elif any(isinstance(x, (list, dict, tuple, set, bytes)) for x in df[col].head(20)):
            problematic.append((col, str(dtype), "contains complex objects"))
    if problematic:
        msg = "The following columns have unsupported or mixed types and may cause display/serialization errors: "
        msg += ", ".join([f"'{col}' (dtype={dtype}, inferred={inferred})" for col, dtype, inferred in problematic])
        return False, msg, problematic
    return True, "DataFrame is valid for Streamlit display.", [] 