"""
ALPHA - End-to-End Machine Learning Platform
Hugging Face Integration Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space, display_code_block
)
from utils.huggingface import (
    check_hf_availability, get_hf_models, run_inference_api, load_hf_model,
    create_hf_pipeline, save_hf_model_info, display_model_card
)
from utils.imports import is_package_available, logger, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Hugging Face")

# Display sidebar navigation
sidebar_navigation()

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for Hugging Face integration. Please install pandas.')
    st.stop()
if not is_package_available('numpy'):
    st.warning('NumPy is not available. Some features may not work.')

# Main content
page_header(
    title="Hugging Face Integration",
    description="Search, use, and integrate Hugging Face models",
    icon="🤗"
)

# Check if API key is properly set
with st.sidebar:
    st.markdown("## 🤗 Hugging Face")
    
    transformers_available = is_package_available("transformers")
    if not transformers_available:
        st.warning("Transformers library not installed. Some features will be limited.")
    
    # Check HF availability with improved function
    hf_status = check_hf_availability()
    if hf_status["status"] == "not_available":
        st.error("Hugging Face API connection failed. Please enter a valid API key below.")
        
        # Add temporary API key input
        st.markdown("### Enter API Key")
        temp_api_key = st.text_input("Enter your Hugging Face API key:", type="password", 
                              help="Your API key will only be used for this session and won't be stored permanently.")
        
        if temp_api_key:
            # Store in session state
            st.session_state["temp_huggingface_api_key"] = temp_api_key
            
            # Test if the key works
            import os
            os.environ["HUGGINGFACE_API_KEY"] = temp_api_key
            
            # Recheck availability with new key
            updated_status = check_hf_availability()
            if updated_status["api_working"]:
                st.success("✅ API key is valid and working! Refresh the page or use the search button to fetch models.")
            else:
                st.error("❌ API key verification failed. Please check that your key is valid and has the correct permissions.")
        
        # Instructions to get API key
        with st.expander("How to get an API key"):
            st.markdown("""
            1. Go to [huggingface.co](https://huggingface.co) and create an account
            2. Go to your profile > Settings > Access Tokens
            3. Create a new token with "Read" permission
            4. Copy and paste the token above
            
            **Note:** For permanent usage, add the API key to your environment variables:
            ```
            HUGGINGFACE_API_KEY=your_key_here
            ```
            """)
    elif hf_status["status"] == "api_only":
        st.success("✅ Hugging Face API is connected")
        st.info("Using API-only mode. Install transformers package for more features.")
    else:
        st.success("✅ Hugging Face API and transformers library are fully available")
        
        # Allow changing API key even when connected
        with st.expander("Change API Key"):
            new_api_key = st.text_input("Enter a different API key:", type="password")
            if new_api_key and st.button("Update API Key"):
                st.session_state["temp_huggingface_api_key"] = new_api_key
                import os
                os.environ["HUGGINGFACE_API_KEY"] = new_api_key
                st.success("API key updated. Please refresh or search again.")
    
    st.markdown("### Tasks")
    task_type = st.selectbox(
        "Select task type",
        options=[
            "text-classification", 
            "token-classification", 
            "question-answering",
            "summarization", 
            "translation", 
            "text-generation", 
            "fill-mask",
            "image-classification",
            "object-detection", 
            "image-segmentation"
        ]
    )

# Create tabs for different functionality
hf_tabs = create_tab_panels(
    "Model Explorer", "Text Generation", "Text Classification", "Image Processing"
)

# Tab 1: Model Explorer
with hf_tabs[0]:
    st.markdown("### 🔍 Search Hugging Face Models")
    
    search_container = st.container()
    with search_container:
        # Search interface
        search_query = st.text_input("Search for models", placeholder="e.g., text classification, BERT, GPT")
        search_limit = st.slider("Number of results", 5, 50, 20)
        search_task = st.checkbox("Filter by selected task", value=True)
        
        search_button = st.button("Search Models")
        
        if search_button or "search_results" in st.session_state:
            with st.spinner("Searching for models..."):
                # Check if we have a valid API key
                api_key = None
                if st.session_state.get("temp_huggingface_api_key"):
                    api_key = st.session_state.get("temp_huggingface_api_key")
                
                # Perform search
                if search_task:
                    filter_task = task_type
                else:
                    filter_task = None
                
                # Pass search query to the get_hf_models function
                models = get_hf_models(task=filter_task, query=search_query, limit=search_limit)
                
                if isinstance(models, dict) and "error" in models:
                    st.error(f"Error: {models['error']} - {models.get('message', '')}")
                    
                    # If we got an authentication error, suggest using a temporary API key
                    if "Authentication" in models.get('error', '') or "Authentication" in models.get('message', ''):
                        st.info("Please check your API key in the sidebar. You can enter a temporary key there.")
                elif not models:
                    st.info("No models found matching your criteria.")
                else:
                    st.session_state["search_results"] = models
                    st.success(f"Found {len(models)} models")
    
    # Display search results
    if "search_results" in st.session_state:
        models = st.session_state["search_results"]
        
        # Display models as a selection list
        st.markdown("### Search Results")
        
        # Convert to DataFrame for display
        models_df = pd.DataFrame([
            {
                "Model ID": model.get("modelId", "N/A"),
                "Pipeline Tag": ", ".join(model.get("pipeline_tag", ["N/A"])),
                "Downloads": model.get("downloads", 0),
                "Author": model.get("author", "N/A")
            }
            for model in models
        ])
        
        is_valid, msg, problematic = validate_dataframe_for_streamlit(models_df)
        if not is_valid:
            st.error(f"Cannot display DataFrame: {msg}")
        else:
            st.dataframe(models_df)
        
        # Model selection
        selected_model_id = st.selectbox(
            "Select a model to view details",
            options=[model.get("modelId", "N/A") for model in models]
        )
        
        if selected_model_id:
            # Display model information
            selected_model = next((m for m in models if m.get("modelId") == selected_model_id), None)
            
            if selected_model:
                with st.expander("Model Details", expanded=True):
                    st.markdown(f"### {selected_model.get('modelId')}")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Author:** {selected_model.get('author', 'N/A')}")
                        st.markdown(f"**Downloads:** {selected_model.get('downloads', 0):,}")
                    
                    with col2:
                        st.markdown(f"**Tasks:** {', '.join(selected_model.get('pipeline_tag', ['N/A']))}")
                        st.markdown(f"**Last Modified:** {selected_model.get('lastModified', 'N/A')}")
                    
                    st.markdown("### Description")
                    st.markdown(selected_model.get('description', 'No description available'))
                
                # Add options to use the model
                st.markdown("### Use This Model")
                
                use_options = st.radio(
                    "How would you like to use this model?",
                    options=["API Inference", "Save for Later", "Code Sample"]
                )
                
                if use_options == "API Inference":
                    st.markdown("#### API Inference")
                    st.info("Try this model through the Hugging Face API")
                    
                    # Select the tab based on the model type
                    if "text-generation" in selected_model.get("pipeline_tag", []):
                        st.session_state["selected_generation_model"] = selected_model_id
                        st.info("Go to the Text Generation tab to try this model")
                    
                    elif "text-classification" in selected_model.get("pipeline_tag", []):
                        st.session_state["selected_classification_model"] = selected_model_id
                        st.info("Go to the Text Classification tab to try this model")
                    
                    elif any(t in selected_model.get("pipeline_tag", []) for t in ["image-classification", "object-detection", "image-segmentation"]):
                        st.session_state["selected_image_model"] = selected_model_id
                        st.info("Go to the Image Processing tab to try this model")
                
                elif use_options == "Save for Later":
                    st.markdown("#### Save Model Reference")
                    
                    model_name = st.text_input("Model Name (for reference)", value=selected_model_id)
                    model_description = st.text_area("Description", value=selected_model.get('description', ''))
                    
                    if st.button("Save Model Reference"):
                        with st.spinner("Saving model reference..."):
                            # Save model reference
                            result = save_hf_model_info(
                                model_name=selected_model_id,
                                task=", ".join(selected_model.get("pipeline_tag", [])),
                                metadata={
                                    "name": model_name,
                                    "description": model_description,
                                    "author": selected_model.get("author", "N/A"),
                                    "pipeline_tag": selected_model.get("pipeline_tag", [])
                                }
                            )
                            
                            if "error" in result:
                                st.error(f"Error saving model reference: {result['error']}")
                            else:
                                st.success(f"Model reference saved with ID: {result['id']}")
                
                elif use_options == "Code Sample":
                    st.markdown("#### Code Sample")
                    
                    code_sample = f"""
# Install dependencies if needed
# !pip install transformers

from transformers import pipeline

# Initialize pipeline with the model
pipe = pipeline("{task_type}", model="{selected_model_id}")

# Example usage
result = pipe("Your input text here")
print(result)
"""
                    
                    display_code_block(code_sample)
                    
                    # Additional sample for inference API
                    api_code = f"""
# Using the Inference API
import requests

API_URL = "https://api-inference.huggingface.co/models/{selected_model_id}"
headers = {{"Authorization": "Bearer $HUGGINGFACE_API_KEY"}}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

result = query({{"inputs": "Your input text here"}})
print(result)
"""
                    
                    display_code_block(api_code)

# Tab 2: Text Generation
with hf_tabs[1]:
    st.markdown("### 🤖 Text Generation Models")
    
    # Use model selected from explorer if available
    default_model = "gpt2"
    if "selected_generation_model" in st.session_state:
        default_model = st.session_state["selected_generation_model"]
    
    # Model selection
    gen_model = st.text_input("Model name", value=default_model)
    
    # Input for text generation
    prompt = st.text_area("Enter prompt", height=150)
    
    # Generation parameters
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Max tokens", 10, 500, 100)
    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    
    # Check if we have a valid API key to pass
    api_key = None
    if st.session_state.get("temp_huggingface_api_key"):
        api_key = st.session_state.get("temp_huggingface_api_key")
    
    # Generate button
    if st.button("Generate Text") and prompt:
        with st.spinner(f"Generating text using {gen_model}..."):
            # Format API request parameters
            parameters = {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "return_full_text": False,
            }
            
            # Call the Hugging Face Inference API
            result = run_inference_api(
                model_name=gen_model,
                inputs=prompt,
                task="text-generation",
                api_key=api_key
            )
            
            # Display results
            if isinstance(result, dict) and "error" in result:
                st.error(f"Error: {result['error']} - {result.get('message', '')}")
            else:
                st.success("Generation complete!")
                
                # Format the result for display
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        generated_text = result[0]["generated_text"]
                    else:
                        generated_text = str(result[0])
                else:
                    generated_text = str(result)
                
                # Display generated text in a box
                st.markdown("### Generated Output")
                st.markdown(f"**Prompt:** {prompt[:50]}...")
                st.text_area("Generated Text", value=generated_text, height=300)
                
                # Add option to save the result
                st.download_button(
                    label="Save text",
                    data=generated_text,
                    file_name=f"generated_text_{gen_model.replace('/', '_')}.txt",
                    mime="text/plain",
                )

# Tab 3: Text Classification
with hf_tabs[2]:
    st.markdown("### 📊 Text Classification")
    
    # Use model selected from explorer if available
    default_classification_model = "distilbert-base-uncased-finetuned-sst-2-english"
    if "selected_classification_model" in st.session_state:
        default_classification_model = st.session_state["selected_classification_model"]
    
    # Model selection
    class_model = st.text_input("Classification model name", value=default_classification_model)
    
    # Input for classification
    text_input = st.text_area("Enter text to classify", height=150)
    
    # Check if we have a valid API key to pass
    api_key = None
    if st.session_state.get("temp_huggingface_api_key"):
        api_key = st.session_state.get("temp_huggingface_api_key")
    
    # Classify button
    if st.button("Classify Text") and text_input:
        with st.spinner(f"Classifying text using {class_model}..."):
            # Call the Hugging Face Inference API
            result = run_inference_api(
                model_name=class_model,
                inputs=text_input,
                task="text-classification",
                api_key=api_key
            )
            
            # Display results
            if isinstance(result, dict) and "error" in result:
                st.error(f"Error: {result['error']} - {result.get('message', '')}")
            else:
                st.success("Classification complete!")
                
                # Format the results for display
                if isinstance(result, list) and len(result) > 0:
                    # Convert results to DataFrame for display
                    if isinstance(result[0], list):
                        # Multiple classification results
                        all_results = []
                        for item in result[0]:
                            all_results.append({
                                "Label": item.get("label", ""),
                                "Score": item.get("score", 0)
                            })
                        results_df = pd.DataFrame(all_results)
                    else:
                        # Single classification result
                        results_df = pd.DataFrame([
                            {"Label": item.get("label", ""), "Score": item.get("score", 0)}
                            for item in result
                        ])
                    
                    # Sort by score
                    results_df = results_df.sort_values(by=["Score"], ascending=False)
                    
                    # Display results
                    st.markdown("### Classification Results")
                    is_valid, msg, problematic = validate_dataframe_for_streamlit(results_df)
                    if not is_valid:
                        st.error(f"Cannot display DataFrame: {msg}")
                    else:
                        st.dataframe(results_df)
                    
                    # Visualize results
                    if not results_df.empty and len(results_df) > 1:
                        st.markdown("### Visualization")
                        import altair as alt
                        chart = alt.Chart(results_df).mark_bar().encode(
                            x=alt.X("Score:Q", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("Label:N", sort="-x"),
                            color=alt.Color("Score:Q", scale=alt.Scale(scheme="viridis"))
                        ).properties(height=30 * len(results_df))
                        st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No classification results returned. The model might not be compatible with text classification.")

# Tab 4: Image Processing
with hf_tabs[3]:
    st.markdown("### 🖼️ Image Processing")
    
    # Model selection for image processing
    image_models = [
        "google/vit-base-patch16-224",
        "facebook/detr-resnet-50",
        "facebook/detr-resnet-101",
        "google/owlvit-base-patch32",
    ]
    
    if "selected_image_model" in st.session_state:
        if st.session_state["selected_image_model"] not in image_models:
            image_models.append(st.session_state["selected_image_model"])
        
        selected_image_model = st.selectbox(
            "Select an image processing model",
            options=image_models,
            index=image_models.index(st.session_state["selected_image_model"])
        )
    else:
        selected_image_model = st.selectbox(
            "Select an image processing model",
            options=image_models
        )
    
    # Display model info
    display_model_card(selected_image_model)
    
    # Select task
    image_task = st.selectbox(
        "Select image task",
        options=["image-classification", "object-detection", "image-segmentation"]
    )
    
    # Upload image
    st.markdown("### Upload Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Process image button
        if st.button("Process Image"):
            with st.spinner("Processing image..."):
                st.markdown("### Processing with API")
                st.info("This feature uses the Hugging Face API to process images. The model will analyze the uploaded image based on the selected task.")
                
                # In a real implementation, you would encode the image and send it to the API
                # For now, we'll just show a placeholder
                st.warning("Image processing via API is a premium feature and requires proper image encoding. Please check the Hugging Face documentation for details.")
                
                # Display sample code
                sample_code = f"""
# Example code for image processing with Hugging Face
from transformers import pipeline
import requests
from PIL import Image

# Load image
image = Image.open("your_image.jpg")

# Initialize pipeline for {image_task}
pipe = pipeline("{image_task}", model="{selected_image_model}")

# Process image
result = pipe(image)
print(result)
"""
                
                display_code_block(sample_code)
                
                # Additional explanation about costs
                st.info("Note: Processing large images with complex models may incur costs on the Hugging Face API.")

# Footer
st.markdown("---")
st.markdown("Powered by 🤗 Hugging Face")
display_info_box(
    text="**About Hugging Face Integration:** This module allows you to search, test, and use state-of-the-art machine learning models from Hugging Face. You can experiment with different models for text generation, classification, and image processing.",
    type="info"
) 