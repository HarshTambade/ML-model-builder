"""
ALPHA - End-to-End Machine Learning Platform
Deployment Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import io
import zipfile
from datetime import datetime
from pathlib import Path

# Import utility modules
from utils.config import MODELS_DIR, get_available_models
from utils.models import load_model, generate_deployment_code
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, display_code_block, add_vertical_space
)
from utils.imports import is_package_available, logger, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Deployment")

# Display sidebar navigation
sidebar_navigation()

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for deployment features. Please install pandas.')
    st.stop()
if not is_package_available('numpy'):
    st.warning('NumPy is not available. Some deployment features may not work.')

# Main content
page_header(
    title="Deployment",
    description="Deploy your trained models with one click",
    icon="🚀"
)

# Function to create a deployment package
def create_deployment_package(model_info, deployment_type):
    """Create a deployment package for the selected model."""
    # Initialize a BytesIO object for the zip file
    zip_buffer = io.BytesIO()
    
    # Create a ZipFile object
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add model pickle file
        model_path = os.path.join(model_info["path"], "model.pkl")
        if os.path.exists(model_path):
            zipf.write(model_path, arcname="model.pkl")
        
        # Add metadata as JSON
        metadata_str = json.dumps(model_info["metadata"], indent=2)
        zipf.writestr("metadata.json", metadata_str)
        
        # Add README file
        readme_content = f"""# {model_info["metadata"]["name"]} Deployment Package

## Model Information
- **Name**: {model_info["metadata"]["name"]}
- **Type**: {model_info["metadata"]["type"]}
- **Created**: {model_info["metadata"]["created_at"]}

## Deployment Instructions

This package contains everything you need to deploy the model.

1. Unzip the package to your deployment directory
2. Install the required packages: `pip install -r requirements.txt`
3. Run the application using the appropriate command for your deployment type
   - For FastAPI: `uvicorn app:app --reload`
   - For standalone: `python predict.py`

## Model Performance

```
{json.dumps(model_info["metadata"].get("performance", {}), indent=2)}
```

## Contact

If you have any questions, please contact the model creator.
"""
        zipf.writestr("README.md", readme_content)
        
        # Add appropriate deployment code
        if deployment_type == "FastAPI":
            # Add app.py
            app_code = generate_deployment_code(model_info, file_type="fastapi")
            zipf.writestr("app.py", app_code)
            
            # Add requirements.txt
            requirements = generate_deployment_code(model_info, file_type="requirements")
            zipf.writestr("requirements.txt", requirements)
            
            # Add Dockerfile
            dockerfile = generate_deployment_code(model_info, file_type="docker")
            zipf.writestr("Dockerfile", dockerfile)
            
        elif deployment_type == "Python Script":
            # Add predict.py
            script_code = generate_deployment_code(model_info, file_type="python")
            zipf.writestr("predict.py", script_code)
            
            # Add simplified requirements.txt
            requirements = """
pandas>=1.5.3
numpy>=1.24.3
scikit-learn>=1.2.2
"""
            zipf.writestr("requirements.txt", requirements)
        
        # Add example data file if available
        example_data = """
# Example data for prediction
# Replace this with your actual data format

feature_1,feature_2,feature_3
1.2,3.4,5.6
7.8,9.0,1.2
"""
        zipf.writestr("example_data.csv", example_data)
    
    # Seek to the beginning of the buffer
    zip_buffer.seek(0)
    return zip_buffer

# Main deployment interface
st.markdown("### Deploy Your Models")

# Get available models
models = get_available_models()

if not models:
    st.warning("No trained models available. Please train a model in the Model Training module.")
else:
    # Model selection
    model_names = [model["name"] for model in models]
    selected_model_name = st.selectbox("Select a model to deploy", model_names)
    
    # Find the selected model
    selected_model_info = next(
        (model for model in models if model["name"] == selected_model_name),
        None
    )
    
    if selected_model_info:
        # Extract model ID from path
        model_path = Path(selected_model_info["path"])
        model_id = model_path.name
        
        # Load the model details
        model_info = load_model(model_id)
        
        if model_info:
            # Display model info
            st.markdown(f"### {model_info['metadata']['name']}")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Type", model_info["metadata"]["type"].capitalize())
            col2.metric("Created", model_info["metadata"]["created_at"])
            
            # Get a performance metric to display
            perf = model_info["metadata"].get("performance", {}).get("metrics", {})
            if perf:
                metric_name = next(iter(perf.keys()))
                metric_value = perf[metric_name]
                col3.metric(metric_name.upper(), f"{metric_value:.4f}")
            
            # Display model details
            with st.expander("Model Details", expanded=False):
                st.json(model_info["metadata"])
            
            # Deployment options
            st.markdown("### Deployment Options")
            
            deployment_tabs = create_tab_panels("Generate Code", "Export Package", "API Documentation")
            
            # Tab 1: Generate Code
            with deployment_tabs[0]:
                st.markdown("#### Generate Deployment Code")
                
                code_type = st.selectbox(
                    "Select code type",
                    ["Python Script", "FastAPI", "Docker"],
                    index=0
                )
                
                # Generate code based on type
                if code_type == "Python Script":
                    code = generate_deployment_code(model_info, file_type="python")
                    st.markdown("##### Python Prediction Script")
                    display_code_block(code, language="python")
                
                elif code_type == "FastAPI":
                    code = generate_deployment_code(model_info, file_type="fastapi")
                    st.markdown("##### FastAPI Application")
                    display_code_block(code, language="python")
                
                elif code_type == "Docker":
                    code = generate_deployment_code(model_info, file_type="docker")
                    st.markdown("##### Dockerfile")
                    display_code_block(code, language="dockerfile")
                
                # Copy code button
                st.download_button(
                    label="Download Code",
                    data=code,
                    file_name=f"{selected_model_name}_{code_type.lower().replace(' ', '_')}.{'py' if code_type != 'Docker' else 'dockerfile'}",
                    mime="text/plain"
                )
            
            # Tab 2: Export Package
            with deployment_tabs[1]:
                st.markdown("#### Export Deployment Package")
                
                package_type = st.selectbox(
                    "Select package type",
                    ["Python Script", "FastAPI"],
                    index=0
                )
                
                st.markdown("""
                The deployment package will include:
                - The trained model
                - Metadata and documentation
                - Deployment code
                - Requirements file
                - Example data and usage
                """)
                
                # Generate package
                if st.button("Generate Deployment Package"):
                    with st.spinner("Creating deployment package..."):
                        # Create the package
                        zip_buffer = create_deployment_package(model_info, package_type)
                        
                        # Offer download
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        filename = f"{selected_model_name}_{package_type.lower().replace(' ', '_')}_{timestamp}.zip"
                        
                        st.success("Deployment package created successfully!")
                        st.download_button(
                            label="Download Package",
                            data=zip_buffer,
                            file_name=filename,
                            mime="application/zip"
                        )
            
            # Tab 3: API Documentation
            with deployment_tabs[2]:
                st.markdown("#### API Documentation")
                
                st.markdown("""
                ### REST API Usage
                
                When deployed as a FastAPI application, your model will expose the following endpoints:
                
                #### GET /
                
                Home endpoint that returns basic information about the model.
                
                **Response:**
                ```json
                {
                    "message": "Welcome to the Model API",
                    "model_type": "classification"  // or regression, etc.
                }
                ```
                
                #### POST /predict
                
                Make predictions with the model.
                
                **Request:**
                ```json
                {
                    "features": [
                        {"feature_1": 1.2, "feature_2": 3.4, "feature_3": 5.6},
                        {"feature_1": 7.8, "feature_2": 9.0, "feature_3": 1.2}
                    ]
                }
                ```
                
                **Response:**
                ```json
                {
                    "predictions": [0, 1]  // or continuous values for regression
                }
                ```
                
                ### Python SDK Usage
                
                ```python
                import requests
                import json
                
                # Prepare data
                data = {
                    "features": [
                        {"feature_1": 1.2, "feature_2": 3.4, "feature_3": 5.6},
                        {"feature_1": 7.8, "feature_2": 9.0, "feature_3": 1.2}
                    ]
                }
                
                # Make prediction request
                response = requests.post(
                    "http://localhost:8000/predict",
                    data=json.dumps(data),
                    headers={"Content-Type": "application/json"}
                )
                
                # Parse results
                result = response.json()
                print(f"Predictions: {result['predictions']}")
                ```
                """)
        else:
            st.error("Error loading model. The model might be corrupted.")

# Deployment guides section
st.markdown("---")
st.markdown("### Deployment Guides")

deployment_guides = create_tab_panels("Local Deployment", "Cloud Deployment", "Model Serving")

# Tab 1: Local Deployment
with deployment_guides[0]:
    st.markdown("""
    ### Local Deployment Guide
    
    #### Option 1: Standalone Python Script
    
    1. **Download** the Python Script package from the "Export Package" tab
    2. **Unzip** the package to your preferred directory
    3. **Install Requirements**:
       ```bash
       pip install -r requirements.txt
       ```
    4. **Run the Script**:
       ```bash
       python predict.py
       ```
    5. **Modify** the script to fit your specific use case
    
    #### Option 2: FastAPI Application
    
    1. **Download** the FastAPI package from the "Export Package" tab
    2. **Unzip** the package to your preferred directory
    3. **Install Requirements**:
       ```bash
       pip install -r requirements.txt
       ```
    4. **Run the Application**:
       ```bash
       uvicorn app:app --reload --host 0.0.0.0 --port 8000
       ```
    5. **Access the API**: Open `http://localhost:8000/docs` in your browser
    
    #### Option 3: Docker Container
    
    1. **Download** the FastAPI package from the "Export Package" tab
    2. **Unzip** the package to your preferred directory
    3. **Build the Docker Image**:
       ```bash
       docker build -t mymodel:latest .
       ```
    4. **Run the Container**:
       ```bash
       docker run -p 8000:8000 mymodel:latest
       ```
    5. **Access the API**: Open `http://localhost:8000/docs` in your browser
    """)

# Tab 2: Cloud Deployment
with deployment_guides[1]:
    st.markdown("""
    ### Cloud Deployment Guide
    
    #### AWS Deployment
    
    1. **Create EC2 Instance**:
       - Choose an Ubuntu Server AMI
       - Select an instance type (t2.micro for testing, larger for production)
       - Configure security groups to allow HTTP traffic (port 80) and SSH (port 22)
    
    2. **Connect to Instance and Set Up**:
       ```bash
       ssh -i your-key.pem ubuntu@your-instance-ip
       sudo apt update && sudo apt upgrade -y
       sudo apt install -y python3-pip docker.io
       sudo systemctl start docker
       sudo systemctl enable docker
       ```
    
    3. **Deploy Your Model**:
       - Upload your deployment package to the instance
       - Unzip and build the Docker image
       - Run the container
    
    #### Google Cloud Platform
    
    1. **Create a VM Instance**:
       - Choose a Compute Engine instance
       - Select an appropriate machine type
       - Use a Debian/Ubuntu image
    
    2. **Deploy with Google Cloud Run** (Serverless):
       - Build and push your Docker image to Google Container Registry
       - Deploy to Cloud Run
       ```bash
       gcloud builds submit --tag gcr.io/PROJECT-ID/mymodel
       gcloud run deploy --image gcr.io/PROJECT-ID/mymodel --platform managed
       ```
    
    #### Microsoft Azure
    
    1. **Deploy with Azure App Service**:
       - Create an App Service with Python support
       - Set up GitHub Actions for CI/CD
       - Configure environment variables
    
    2. **Deploy with Azure Container Instances**:
       - Build and push your Docker image to Azure Container Registry
       - Deploy to Azure Container Instances
    """)

# Tab 3: Model Serving
with deployment_guides[2]:
    st.markdown("""
    ### Advanced Model Serving Guide
    
    #### MLflow Deployment
    
    [MLflow](https://mlflow.org/) is an open-source platform for the complete machine learning lifecycle.
    
    1. **Install MLflow**:
       ```bash
       pip install mlflow
       ```
    
    2. **Log Model with MLflow**:
       ```python
       import mlflow
       
       # Log the model
       with mlflow.start_run():
           # Log model parameters
           mlflow.log_params(params)
           
           # Log model metrics
           mlflow.log_metrics(metrics)
           
           # Log the model
           mlflow.sklearn.log_model(model, "model")
       ```
    
    3. **Serve the Model**:
       ```bash
       mlflow models serve -m runs:/<run-id>/model
       ```
    
    #### TensorFlow Serving
    
    For TensorFlow models, [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) is a flexible deployment solution.
    
    1. **Save Model in SavedModel Format**:
       ```python
       import tensorflow as tf
       
       # Save the model
       tf.saved_model.save(model, "saved_model_dir")
       ```
    
    2. **Run TensorFlow Serving**:
       ```bash
       docker pull tensorflow/serving
       
       docker run -p 8501:8501 \\
         --mount type=bind,source=/path/to/saved_model_dir,target=/models/mymodel \\
         -e MODEL_NAME=mymodel -t tensorflow/serving
       ```
    
    #### Kubernetes Deployment
    
    For production environments, [Kubernetes](https://kubernetes.io/) offers scalable and resilient deployment options.
    
    1. **Create a Kubernetes Deployment**:
       ```yaml
       apiVersion: apps/v1
       kind: Deployment
       metadata:
         name: model-deployment
       spec:
         replicas: 3
         selector:
           matchLabels:
             app: model-server
         template:
           metadata:
             labels:
               app: model-server
           spec:
             containers:
             - name: model-container
               image: mymodel:latest
               ports:
               - containerPort: 8000
       ```
    
    2. **Create a Service**:
       ```yaml
       apiVersion: v1
       kind: Service
       metadata:
         name: model-service
       spec:
         selector:
           app: model-server
         ports:
         - port: 80
           targetPort: 8000
         type: LoadBalancer
       ```
    
    3. **Apply the Configuration**:
       ```bash
       kubectl apply -f deployment.yaml
       kubectl apply -f service.yaml
       ```
    """)

# Additional sidebar options
with st.sidebar:
    st.markdown("## 🚀 Deployment Options")
    
    with st.expander("💡 Deployment Tips"):
        st.markdown("""
        ### Best Practices for Model Deployment
        
        1. **Version Control**: Keep track of model versions and deployments
        
        2. **Monitoring**: Implement monitoring to track model performance in production
        
        3. **Scalability**: Design your deployment to handle varying loads
        
        4. **Security**: Secure your API endpoints and manage access appropriately
        
        5. **Testing**: Thoroughly test your deployment before releasing to production
        
        6. **Documentation**: Provide clear documentation for API users
        """)

# Before any DataFrame display or visualization, validate the DataFrame
# Example for main workflow (add similar checks before each st.dataframe, st.table, or visualization):
#
# is_valid, msg, problematic = validate_dataframe_for_streamlit(df)
# if not is_valid:
#     st.error(f"Cannot display DataFrame: {msg}")
# else:
#     st.dataframe(df)
#     # or create visualization 