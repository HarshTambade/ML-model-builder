"""
Script to generate sample datasets for ALPHA platform testing
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import generate_synthetic_data, save_dataset

def main():
    """Generate sample datasets for testing"""
    
    print("Generating sample classification dataset...")
    classification_df = generate_synthetic_data(
        n_samples=500,
        n_features=10,
        data_type="classification",
        n_classes=3,
        n_informative=5,
        class_sep=2.0
    )
    
    print("Saving classification dataset...")
    save_dataset(
        df=classification_df,
        dataset_name="Sample Classification Dataset",
        description="A synthetic classification dataset for testing",
        dataset_type="synthetic"
    )
    
    print("Generating sample regression dataset...")
    regression_df = generate_synthetic_data(
        n_samples=1000,
        n_features=15,
        data_type="regression",
        n_informative=8
    )
    
    print("Saving regression dataset...")
    save_dataset(
        df=regression_df,
        dataset_name="Sample Regression Dataset",
        description="A synthetic regression dataset for testing",
        dataset_type="synthetic"
    )
    
    print("Generating sample clustering dataset...")
    clustering_df = generate_synthetic_data(
        n_samples=800,
        n_features=10,
        data_type="clustering",
        n_classes=4,
        cluster_std=0.8
    )
    
    print("Saving clustering dataset...")
    save_dataset(
        df=clustering_df,
        dataset_name="Sample Clustering Dataset",
        description="A synthetic clustering dataset for testing",
        dataset_type="synthetic"
    )
    
    print("All sample datasets generated successfully!")

if __name__ == "__main__":
    main() 