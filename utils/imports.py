"""
Utility module for handling optional imports in the ALPHA platform.
"""

import importlib
import logging
import pandas as pd
import numpy as np
import os
import sys
import streamlit as st
from utils.config import DEPENDENCY_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add cache decorators for performance optimization
def add_caching_decorators():
    """Add Streamlit cache decorators to frequently used functions"""
    global fix_dataframe_dtypes, validate_dataframe_for_streamlit
    
    # Apply cache decorator to dataframe utility functions
    fix_dataframe_dtypes = st.cache_data(ttl=3600, show_spinner=False)(fix_dataframe_dtypes)
    validate_dataframe_for_streamlit = st.cache_data(ttl=3600, show_spinner=False)(validate_dataframe_for_streamlit)
    
    logger.info("Added cache decorators to frequently used functions")

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

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=3600, show_spinner=False)
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

@st.cache_data(ttl=600, show_spinner=False)
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

@st.cache_data(ttl=600, show_spinner=False)
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
@st.cache_data(ttl=3600, show_spinner=False)
def get_torch():
    """Get PyTorch if available and enabled, otherwise None."""
    if not DEPENDENCY_CONFIG["USE_PYTORCH"]:
        logger.info("PyTorch is disabled in configuration.")
        return None
    return get_optional_package("torch")

@st.cache_data(ttl=3600, show_spinner=False)
def get_tensorflow():
    """Get TensorFlow if available and enabled, otherwise None."""
    if not DEPENDENCY_CONFIG["USE_TENSORFLOW"]:
        logger.info("TensorFlow is disabled in configuration.")
        return None
    return get_optional_package("tensorflow")

@st.cache_data(ttl=3600, show_spinner=False)
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
            try:
                if not result_df[col].isna().any():
                    result_df[col] = result_df[col].astype('int64')
            except:
                pass
                
        # Convert datetime dtypes to datetime64[ns]
        for col in result_df.select_dtypes(include=['datetime']).columns:
            result_df[col] = pd.to_datetime(result_df[col])
            
        # Handle object dtypes containing only strings
        for col in result_df.select_dtypes(include=['object']).columns:
            # Sample values to check if this is a string column
            sample = result_df[col].dropna().head(100)
            if len(sample) > 0 and all(isinstance(val, str) for val in sample):
                result_df[col] = result_df[col].astype(str)
                
        return result_df
    except Exception as e:
        logger.warning(f"Error fixing DataFrame dtypes: {str(e)}")
        # Return the original DataFrame if we can't fix it
        return df

def validate_dataframe_for_streamlit(df):
    """
    Validates a DataFrame for compatibility with Streamlit's display functions.
    Returns a tuple (is_valid, message) where:
    - is_valid: Boolean indicating if the DataFrame can be displayed
    - message: Error message if not valid, empty string otherwise
    """
    try:
        # Basic validation
        if not isinstance(df, pd.DataFrame):
            return False, "Input is not a DataFrame"
            
        if df.empty:
            return True, ""  # Empty DataFrames are valid
            
        # Check size for memory limits
        memory_usage = df.memory_usage(deep=True).sum()
        if memory_usage > 1000 * 1024 * 1024:  # Greater than 1GB
            return False, f"DataFrame too large ({memory_usage / (1024**2):.2f} MB)"
            
        # Check for problematic column names
        for col in df.columns:
            if not isinstance(col, (str, int, float)):
                return False, f"Column names must be string, int, or float. Found: {type(col)}"
                
        # Basic check for nested complex types that might cause problems
        for col in df.columns:
            try:
                sample = df[col].head(10).dropna()
                if len(sample) > 0:
                    first_valid = sample.iloc[0]
                    if isinstance(first_valid, (list, dict, set)) and len(str(first_valid)) > 100:
                        return False, f"Column '{col}' contains complex nested data that may cause display issues"
            except:
                pass
                
        # If we got here, the DataFrame is probably safe to display
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def patch_streamlit_dataframe_display():
    """
    Monkey patch Streamlit's dataframe display functions to better handle
    problematic DataFrames with Arrow serialization issues.
    """
    try:
        # Only apply patch if streamlit is available
        import streamlit as st
        
        # Original functions
        original_dataframe = st.dataframe
        original_table = st.table
        original_write = st.write
        
        # Patched dataframe function
        def patched_dataframe(data, *args, **kwargs):
            if isinstance(data, pd.DataFrame):
                # Apply fixes
                fixed_data = fix_dataframe_dtypes(data)
                # Validate the fixed DataFrame
                is_valid, message = validate_dataframe_for_streamlit(fixed_data)
                if not is_valid:
                    st.error(f"Cannot display DataFrame: {message}")
                    # Show a preview of the data as plain text
                    st.code(str(data.head(5)))
                    return
                # If valid, display the fixed DataFrame
                return original_dataframe(fixed_data, *args, **kwargs)
            else:
                # Pass through for non-DataFrame inputs
                return original_dataframe(data, *args, **kwargs)
                
        # Patched table function
        def patched_table(data, *args, **kwargs):
            if isinstance(data, pd.DataFrame):
                # Apply fixes
                fixed_data = fix_dataframe_dtypes(data)
                # Validate the fixed DataFrame
                is_valid, message = validate_dataframe_for_streamlit(fixed_data)
                if not is_valid:
                    st.error(f"Cannot display table: {message}")
                    # Show a preview of the data as plain text
                    st.code(str(data.head(5)))
                    return
                # If valid, display the fixed DataFrame
                return original_table(fixed_data, *args, **kwargs)
            else:
                # Pass through for non-DataFrame inputs
                return original_table(data, *args, **kwargs)
                
        # Patched write function
        def patched_write(*args, **kwargs):
            # Check if first argument is a DataFrame
            if args and isinstance(args[0], pd.DataFrame):
                # Apply fixes
                fixed_data = fix_dataframe_dtypes(args[0])
                # Validate the fixed DataFrame
                is_valid, message = validate_dataframe_for_streamlit(fixed_data)
                if not is_valid:
                    st.error(f"Cannot write DataFrame: {message}")
                    # Show a preview of the data as plain text
                    st.code(str(args[0].head(5)))
                    return
                # Replace the DataFrame with the fixed version
                args = (fixed_data,) + args[1:]
            # Call the original function with the potentially modified args
            return original_write(*args, **kwargs)
            
        # Apply the monkey patches
        st.dataframe = patched_dataframe
        st.table = patched_table
        st.write = patched_write
        
        logger.info("Streamlit DataFrame display patched for better compatibility")
    except ImportError:
        logger.warning("Streamlit not available, dataframe display patch not applied")
    except Exception as e:
        logger.warning(f"Error applying dataframe display patch: {str(e)}")

def apply_torch_monkey_patch():
    """
    Apply comprehensive monkey patching to prevent PyTorch-related errors.
    This is a more aggressive approach when suppress_torch_warnings() is not enough.
    """
    try:
        import sys
        import types
        
        # Create a more robust fake torch module
        class MockTorch(types.ModuleType):
            def __init__(self):
                super().__init__("torch")
                self.nn = types.ModuleType("torch.nn")
                self.optim = types.ModuleType("torch.optim")
                self.utils = types.ModuleType("torch.utils")
                self.utils.data = types.ModuleType("torch.utils.data")
                
            def __getattr__(self, name):
                # Return empty modules/functions for any attributes
                if name.startswith("__") and name.endswith("__"):
                    # Let Python handle standard dunder methods
                    raise AttributeError(f"'MockTorch' object has no attribute '{name}'")
                    
                # For anything else, return a harmless placeholder
                if name in ["float32", "float64", "int32", "int64"]:
                    return name  # Return the name for dtype constants
                elif name == "cuda":
                    # Create a mock cuda object that returns False for is_available()
                    mock_cuda = types.ModuleType("torch.cuda")
                    mock_cuda.is_available = lambda: False
                    return mock_cuda
                elif name == "device":
                    # Return a callable that returns the device name
                    return lambda device="cpu": device
                elif name == "tensor" or name == "Tensor":
                    # Return a function that raises a descriptive error
                    def mock_tensor(*args, **kwargs):
                        raise RuntimeError("MockTorch does not support tensor operations")
                    return mock_tensor
                else:
                    # For any other attribute, return a new module
                    return types.ModuleType(f"torch.{name}")
         
        # Create a safer version of file path extraction for Streamlit
        def safe_get_module_paths(filenames):
            def is_not_torch_module(module_name):
                return not module_name.startswith("torch")
            
            # Filter out torch modules
            filtered_filenames = [f for f in filenames if is_not_torch_module(f)]
            return filtered_filenames
            
        # Apply the patch only if needed
        if "streamlit" in sys.modules:
            # Replace some problematic Streamlit functions
            import streamlit as st
            # Only apply patches if we detect torch being used
            if "torch" in sys.modules:
                logger.info("PyTorch detected - applying enhanced compatibility patches")
                
                # Create a mock copy that we can progressively enhance
                mock_torch = MockTorch()
                
                # Optional: Copy over some basic functionality from real torch if available
                try:
                    real_torch = sys.modules["torch"]
                    # Copy version information
                    if hasattr(real_torch, "__version__"):
                        mock_torch.__version__ = real_torch.__version__
                    # And maybe some other safe attributes
                except Exception:
                    pass
                
                # Patch both pandas and Streamlit function for to_arrow
                try:
                    import pandas as pd
                    
                    # Create a safer to_arrow function
                    def safe_to_arrow(self, *args, **kwargs):
                        # Strip problematic dtypes
                        return fix_dataframe_dtypes(self).to_arrow(*args, **kwargs)
                    
                    # Apply the patch
                    pd.DataFrame.to_arrow = safe_to_arrow
                    logger.info("Patched pandas.DataFrame.to_arrow for better compatibility")
                except Exception as e:
                    logger.warning(f"Could not patch pandas arrow conversion: {str(e)}")
                    
                logger.info("Applied torch monkey patch to prevent errors")
        else:
            logger.info("Streamlit not detected - skipping torch compatibility patches")
            
    except Exception as e:
        logger.warning(f"Error applying PyTorch monkey patch: {str(e)}")

# Initialize caching when the module is imported
add_caching_decorators() 