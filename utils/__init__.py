"""
Utility modules for the ALPHA ML Platform.

This package includes utility modules for data handling, model management,
UI components, and other helper functions used throughout the platform.
"""

# Import key functions for easy access
from .imports import is_package_available, get_optional_package
from .helpers import load_css, create_download_link, format_large_number, truncate_text

# Initialize configuration
from .config import OPENAI_API_KEY, KAGGLE_USERNAME, KAGGLE_API_KEY

# This file marks the directory as a Python package 