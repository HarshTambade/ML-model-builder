"""
Hugging Face integration utility module for the ALPHA platform.
Contains functions for interacting with Hugging Face models and APIs.
"""

import os
import requests
import json
import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from utils.config import HUGGINGFACE_API_KEY, MODELS_DIR
from utils.imports import is_package_available, get_optional_package, logger

# Set up API key in environment - get most updated value from config each time
def initialize_hf_api_key():
    """Initialize Hugging Face API key in the environment and return its value.
    Will first check for environment variable, then config, then session state.
    """
    # Get API key from environment or config
    api_key = os.environ.get("HUGGINGFACE_API_KEY", "")
    
    # If not in environment, try to get from config
    if not api_key:
        from utils.config import HUGGINGFACE_API_KEY
        api_key = HUGGINGFACE_API_KEY
    
    # If still not found but available in session state (temporary key), use that
    if not api_key and st.session_state.get("temp_huggingface_api_key"):
        api_key = st.session_state.get("temp_huggingface_api_key")
    
    # Set in environment if we found a key
    if api_key:
        os.environ["HUGGINGFACE_API_KEY"] = api_key
        
    return api_key

# Check for transformers library
TRANSFORMERS_AVAILABLE = is_package_available("transformers")
if TRANSFORMERS_AVAILABLE:
    try:
        from transformers import (
            AutoTokenizer, 
            AutoModel, 
            AutoModelForSequenceClassification,
            AutoModelForCausalLM,
            pipeline
        )
    except ImportError as e:
        logger.warning(f"Error importing transformers components: {str(e)}")
        TRANSFORMERS_AVAILABLE = False

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
HUGGINGFACE_HUB_URL = "https://huggingface.co/api/"

def check_hf_availability():
    """Check if Hugging Face integration is available and test API connectivity."""
    api_key = initialize_hf_api_key()
    api_available = bool(api_key)
    
    # Verify API key works if available
    api_working = False
    if api_available:
        try:
            # Simple test request to verify API key works
            headers = {"Authorization": f"Bearer {api_key}"}
            test_response = requests.get(
                f"{HUGGINGFACE_HUB_URL}models?limit=1", 
                headers=headers,
                timeout=5
            )
            api_working = test_response.status_code == 200
            
            if not api_working:
                status_code = test_response.status_code
                logger.warning(f"Hugging Face API key test failed with status: {status_code}")
                
                # Provide more specific information for common error codes
                if status_code == 401:
                    logger.error("Authentication failed with Hugging Face API: 401 - Invalid API key")
                elif status_code == 403:
                    logger.error("Authentication failed with Hugging Face API: 403 - Forbidden, check permissions")
                elif status_code == 429:
                    logger.error("Authentication failed with Hugging Face API: 429 - Rate limit exceeded")
                else:
                    logger.error(f"Authentication failed with Hugging Face API: {status_code}")
        except requests.exceptions.Timeout:
            logger.warning("Timeout when testing Hugging Face API connection")
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error when testing Hugging Face API")
        except Exception as e:
            logger.warning(f"Error testing Hugging Face API: {str(e)}")
    
    return {
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "api_available": api_available,
        "api_working": api_working,
        "status": "fully_available" if TRANSFORMERS_AVAILABLE and api_working else 
                  "api_only" if api_working else 
                  "not_available"
    }

def query_mistral_llm(prompt, max_length=512, api_key=None, model_name=None):
    """
    Query a language model using Hugging Face's Inference API.
    
    Args:
        prompt: The text prompt to send to the model
        max_length: Maximum length of the generated response
        api_key: Optional API key to use instead of default
        model_name: Optional model name to use instead of default Mistral model
        
    Returns:
        Dict containing either the generated text or error information
    """
    try:
        # Get the most up-to-date API key
        api_key = api_key or initialize_hf_api_key()
        
        # Check if API key is available
        if not api_key:
            return {
                "error": "Missing API key",
                "message": "Hugging Face API key is not configured. Please set it in the configuration or environment."
            }
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Use specified model or default to Mistral
        if not model_name or model_name == "Local Text Generation (Rule-Based)":
            model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            model_id = model_name
        
        # Format prompt differently based on the model
        if "mistral" in model_id.lower():
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        elif "flan-t5" in model_id.lower():
            formatted_prompt = prompt
        elif "tinyllama" in model_id.lower() or "llama" in model_id.lower():
            formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        else:
            # Default format for most models
            formatted_prompt = prompt
        
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": 0.7,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        # Make the API request with timeout
        response = requests.post(
            f"{HUGGINGFACE_API_URL}{model_id}",
            headers=headers,
            json=payload,
            timeout=45  # Longer timeout for larger models
        )
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return {"generated_text": result[0].get("generated_text", "").strip()}
            return {"generated_text": "I couldn't generate a proper response. Please try again."}
            
        elif response.status_code == 503:
            logger.warning(f"Hugging Face service unavailable: {response.status_code}")
            return {
                "error": "Service Unavailable",
                "message": "The Hugging Face service is currently unavailable or the model is loading. Please try again in a moment."
            }
        elif response.status_code == 403 or response.status_code == 401:
            logger.error(f"Authentication error: {response.status_code}")
            return {
                "error": "Authentication Failed",
                "message": "Failed to authenticate with Hugging Face API. Please check your API key."
            }
        else:
            logger.error(f"Inference API error: {response.status_code}, {response.text[:100]}")
            return {
                "error": f"API Error: {response.status_code}",
                "message": "Error communicating with Hugging Face API."
            }
    
    except requests.exceptions.Timeout:
        logger.error("Timeout when calling Hugging Face Inference API")
        return {
            "error": "API Timeout",
            "message": f"Request to Hugging Face API for model {model_id} timed out. Large models may take time to load."
        }
    except requests.exceptions.ConnectionError:
        logger.error("Connection error when calling Hugging Face Inference API")
        return {
            "error": "Connection Error",
            "message": "Could not connect to Hugging Face API. Please check your internet connection."
        }
    except Exception as e:
        logger.error(f"Error calling Hugging Face Inference API: {str(e)}")
        return {
            "error": "API Error",
            "message": f"An unexpected error occurred: {str(e)}"
        }

def get_hf_models(task=None, query=None, limit=20):
    """Get a list of available models from Hugging Face for a specific task and search query."""
    try:
        # Get the latest API key
        api_key = initialize_hf_api_key()
        
        # API endpoint URL
        api_url = "https://huggingface.co/api/models"
        
        # Request parameters
        params = {
            "limit": limit,
            "sort": "downloads",
            "direction": -1
        }
        
        # Apply search filters
        if task:
            params["filter"] = task
        
        if query:
            params["search"] = query
        
        # Make the request
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        
        if response.status_code == 200:
            models = response.json()
            return models
        elif response.status_code == 401 or response.status_code == 403:
            logger.error(f"Authentication failed with Hugging Face API: {response.status_code}")
            return {"error": "Authentication failed", "message": "Please check your Hugging Face API key"}
        elif response.status_code == 503:
            logger.error(f"Hugging Face API unavailable: {response.status_code}")
            return {"error": "Hugging Face service is currently unavailable", "message": "Please try again later"}
        else:
            logger.error(f"Failed to fetch models: {response.status_code}")
            return {"error": f"API error: {response.status_code}", "message": response.text}
    
    except Exception as e:
        logger.error(f"Error fetching Hugging Face models: {str(e)}")
        return {"error": str(e)}

def run_inference_api(model_name, inputs, task=None, api_key=None):
    """Run inference using the Hugging Face Inference API."""
    try:
        # Get the latest API key if not provided
        api_key = api_key or initialize_hf_api_key()
        
        if not api_key:
            return {
                "error": "Missing API key",
                "message": "Hugging Face API key is not configured. Please set it in the configuration or environment."
            }
        
        # Check for large models to avoid timeouts
        large_models = [
            "mistralai/Mistral-7B",
            "meta-llama",
            "llama",
            "falcon-40b",
            "falcon-180b",
            "bigcode",
            "codellama"
        ]
        
        # Warn about large models
        if any(lm in model_name.lower() for lm in large_models):
            return {
                "error": "Model too large",
                "message": f"The model {model_name} is too large to run via the inference API. Please select a smaller model."
            }
        
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {"inputs": inputs}
        if task:
            payload["parameters"] = {"task": task}
        
        # Make the API request
        response = requests.post(
            f"{HUGGINGFACE_API_URL}{model_name}",
            headers=headers,
            json=payload,
            timeout=30  # Add timeout
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            logger.error(f"Hugging Face service unavailable: {response.status_code}")
            return {"error": "Hugging Face service is currently unavailable", "message": "Please try again later"}
        elif response.status_code == 403 or response.status_code == 401:
            logger.error(f"Authentication error: {response.status_code}")
            return {"error": "Authentication failed", "message": "Please check your Hugging Face API key"}
        else:
            logger.error(f"Inference API error: {response.status_code}, {response.text[:100]}")
            return {"error": f"API Error: {response.status_code}", "message": "Error communicating with Hugging Face API"}
    
    except requests.exceptions.Timeout:
        logger.error("Timeout when calling Hugging Face Inference API")
        return {"error": "API Timeout", "message": "Request to Hugging Face API timed out"}
    except requests.exceptions.ConnectionError:
        logger.error("Connection error when calling Hugging Face Inference API")
        return {"error": "Connection Error", "message": "Could not connect to Hugging Face API"}
    except Exception as e:
        logger.error(f"Error calling Hugging Face Inference API: {str(e)}")
        return {"error": "API Error", "message": str(e)}

def load_hf_model(model_name, task="text-classification"):
    """Load a Hugging Face model for local inference."""
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "Transformers library not available"}
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model based on task
        if task == "text-classification":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif task == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = AutoModel.from_pretrained(model_name)
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "task": task,
            "name": model_name
        }
    
    except Exception as e:
        logger.error(f"Error loading Hugging Face model: {str(e)}")
        return {"error": str(e)}

def create_hf_pipeline(task, model_name=None):
    """Create a Hugging Face pipeline for inference."""
    if not TRANSFORMERS_AVAILABLE:
        return {"error": "Transformers library not available"}
    
    try:
        # Create pipeline
        if model_name:
            pipe = pipeline(task, model=model_name)
        else:
            pipe = pipeline(task)
        
        return {
            "pipeline": pipe,
            "task": task,
            "name": model_name or "default"
        }
    
    except Exception as e:
        logger.error(f"Error creating Hugging Face pipeline: {str(e)}")
        return {"error": str(e)}

def save_hf_model_info(model_name, task, metadata=None):
    """Save a reference to a Hugging Face model in the platform."""
    try:
        timestamp = int(pd.Timestamp.now().timestamp())
        model_id = f"huggingface_{model_name.replace('/', '_')}_{timestamp}"
        model_dir = MODELS_DIR / model_id
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare metadata
        model_metadata = {
            "name": model_name,
            "type": "huggingface",
            "task": task,
            "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "id": model_id,
            "huggingface_model_name": model_name
        }
        
        # Add user-provided metadata
        if metadata:
            model_metadata.update(metadata)
        
        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)
        
        return {
            "id": model_id,
            "path": str(model_dir),
            "metadata": model_metadata
        }
    
    except Exception as e:
        logger.error(f"Error saving Hugging Face model info: {str(e)}")
        return {"error": str(e)}

def display_model_card(model_name):
    """Display information about a Hugging Face model."""
    try:
        # Get the latest API key
        api_key = initialize_hf_api_key()
        
        # Fetch model information
        api_url = f"https://huggingface.co/api/models/{model_name}"
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.get(api_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            model_info = response.json()
            
            # Display model card in Streamlit
            st.markdown(f"## Model: {model_info.get('modelId', model_name)}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Author:** {model_info.get('author', 'N/A')}")
                st.markdown(f"**Downloads:** {model_info.get('downloads', 'N/A'):,}")
            
            with col2:
                st.markdown(f"**Task:** {', '.join(model_info.get('pipeline_tag', ['N/A']))}")
                st.markdown(f"**Last modified:** {model_info.get('lastModified', 'N/A')}")
            
            st.markdown("### Model Description")
            st.markdown(model_info.get('description', 'No description available'))
            
            return model_info
        else:
            st.error(f"Failed to fetch model information: {response.status_code}")
            return None
    
    except Exception as e:
        st.error(f"Error displaying model card: {str(e)}")
        return None 