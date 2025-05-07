"""
ALPHA - End-to-End Machine Learning Platform
Computer Vision Module
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)
from utils.imports import is_package_available, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Computer Vision")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Computer Vision",
    description="Analyze and process image and video data",
    icon="👁️"
)

# Check for advanced CV libraries
opencv_available = is_package_available("cv2")
sklearn_available = is_package_available("sklearn")
transformers_available = is_package_available("transformers")

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for computer vision features. Please install pandas.')
    st.stop()
if not is_package_available('numpy'):
    st.warning('NumPy is not available. Some features may not work.')

# Create tabs
cv_tabs = create_tab_panels("Image Analysis", "Image Processing", "Image Classification", "Object Detection")

# Helper functions for basic image operations
def get_image_info(image):
    """Get basic information about an image."""
    width, height = image.size
    format_type = image.format if image.format else "Unknown"
    mode = image.mode
    
    # Get color channels
    channels = len(image.getbands())
    
    # Convert to numpy array for additional analysis
    img_array = np.array(image)
    
    # Get basic stats for each channel
    channel_stats = []
    if channels == 1:  # Grayscale
        channel_stats.append({
            "channel": "Gray",
            "min": int(img_array.min()),
            "max": int(img_array.max()),
            "mean": round(float(img_array.mean()), 2),
            "std": round(float(img_array.std()), 2)
        })
    else:  # RGB or RGBA
        for i, channel_name in enumerate(["Red", "Green", "Blue", "Alpha"][:channels]):
            channel_data = img_array[:, :, i]
            channel_stats.append({
                "channel": channel_name,
                "min": int(channel_data.min()),
                "max": int(channel_data.max()),
                "mean": round(float(channel_data.mean()), 2),
                "std": round(float(channel_data.std()), 2)
            })
    
    return {
        "width": width,
        "height": height,
        "format": format_type,
        "mode": mode,
        "channels": channels,
        "channel_stats": channel_stats,
        "file_size_kb": img_array.nbytes / 1024
    }

def apply_basic_processing(image, options):
    """Apply basic image processing operations."""
    result = image.copy()
    
    # Apply operations based on options
    if options.get("grayscale"):
        result = ImageOps.grayscale(result)
    
    if options.get("brightness") != 1.0:
        enhancer = ImageEnhance.Brightness(result)
        result = enhancer.enhance(options.get("brightness"))
    
    if options.get("contrast") != 1.0:
        enhancer = ImageEnhance.Contrast(result)
        result = enhancer.enhance(options.get("contrast"))
    
    if options.get("sharpness") != 1.0:
        enhancer = ImageEnhance.Sharpness(result)
        result = enhancer.enhance(options.get("sharpness"))
    
    if options.get("blur_radius") > 0:
        result = result.filter(ImageFilter.GaussianBlur(radius=options.get("blur_radius")))
    
    if options.get("edge_enhance"):
        result = result.filter(ImageFilter.EDGE_ENHANCE)
    
    if options.get("find_edges"):
        result = result.filter(ImageFilter.FIND_EDGES)
    
    if options.get("emboss"):
        result = result.filter(ImageFilter.EMBOSS)
    
    if options.get("rotate") != 0:
        result = result.rotate(options.get("rotate"), expand=True)
    
    if options.get("flip_horizontal"):
        result = ImageOps.mirror(result)
    
    if options.get("flip_vertical"):
        result = ImageOps.flip(result)
    
    return result

# Function to get image histogram
def get_image_histogram(image):
    """Create histogram visualization for an image."""
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    
    # Get channels
    channels = len(image.getbands())
    
    if channels == 1:  # Grayscale
        img_array = np.array(image)
        axes.hist(img_array.ravel(), bins=256, color='gray', alpha=0.7)
        axes.set_title("Grayscale Histogram")
        axes.set_xlabel("Pixel Value")
        axes.set_ylabel("Frequency")
    else:  # RGB
        img_array = np.array(image)
        colors = ['red', 'green', 'blue']
        
        for i, color in enumerate(colors[:channels]):
            axes.hist(img_array[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=color.capitalize())
        
        axes.set_title("RGB Histogram")
        axes.set_xlabel("Pixel Value")
        axes.set_ylabel("Frequency")
        axes.legend()
    
    plt.tight_layout()
    return fig

# Function to create image grid
def create_image_grid(images, titles):
    """Create a grid of images for comparison."""
    n = len(images)
    if n == 0:
        return None
    
    # Determine grid layout
    if n <= 2:
        cols = n
        rows = 1
    elif n <= 4:
        cols = 2
        rows = (n + 1) // 2
    else:
        cols = 3
        rows = (n + 2) // 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single image case
    if n == 1:
        axes.imshow(np.array(images[0]))
        axes.set_title(titles[0])
        axes.axis('off')
    else:
        # Handle multiple images
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i in range(n):
            axes[i].imshow(np.array(images[i]))
            axes[i].set_title(titles[i])
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(n, len(axes)):
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

# Tab 1: Image Analysis
with cv_tabs[0]:
    st.markdown("### Basic Image Analysis")
    
    # Image input
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        try:
            # Read and display the image
            image = Image.open(uploaded_file)
            
            # Display the image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Get and display image information
            image_info = get_image_info(image)
            
            # Basic image details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Width", f"{image_info['width']}px")
            with col2:
                st.metric("Height", f"{image_info['height']}px")
            with col3:
                st.metric("Channels", image_info['channels'])
            
            # More details
            st.markdown("#### Image Details")
            st.markdown(f"**Format:** {image_info['format']}")
            st.markdown(f"**Mode:** {image_info['mode']}")
            st.markdown(f"**File Size:** {image_info['file_size_kb']:.2f} KB")
            
            # Channel statistics
            st.markdown("#### Channel Statistics")
            
            # Convert channel stats to DataFrame for display
            channel_df = pd.DataFrame(image_info['channel_stats'])
            is_valid, msg, problematic = validate_dataframe_for_streamlit(channel_df)
            if not is_valid:
                st.error(f"Cannot display DataFrame: {msg}")
            else:
                st.dataframe(channel_df)
            
            # Histogram
            st.markdown("#### Pixel Histogram")
            hist_fig = get_image_histogram(image)
            st.pyplot(hist_fig)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Tab 2: Image Processing
with cv_tabs[1]:
    st.markdown("### Image Processing")
    
    if 'uploaded_file' not in locals() or uploaded_file is None:
        st.info("Please upload an image in the Image Analysis tab to use the processing features.")
    else:
        try:
            # Processing options
            st.markdown("#### Processing Options")
            
            # Create columns for options
            col1, col2 = st.columns(2)
            
            with col1:
                # Basic operations
                st.markdown("**Basic Transformations**")
                grayscale = st.checkbox("Convert to Grayscale")
                brightness = st.slider("Brightness", 0.0, 3.0, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.0, 3.0, 1.0, 0.1)
                sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
                blur_radius = st.slider("Blur Radius", 0.0, 10.0, 0.0, 0.5)
            
            with col2:
                # Effects and transformations
                st.markdown("**Effects & Transformations**")
                edge_enhance = st.checkbox("Edge Enhancement")
                find_edges = st.checkbox("Find Edges")
                emboss = st.checkbox("Emboss")
                rotate = st.slider("Rotate (degrees)", -180, 180, 0, 5)
                flip_horizontal = st.checkbox("Flip Horizontal")
                flip_vertical = st.checkbox("Flip Vertical")
            
            # Collect all options
            processing_options = {
                "grayscale": grayscale,
                "brightness": brightness,
                "contrast": contrast,
                "sharpness": sharpness,
                "blur_radius": blur_radius,
                "edge_enhance": edge_enhance,
                "find_edges": find_edges,
                "emboss": emboss,
                "rotate": rotate,
                "flip_horizontal": flip_horizontal,
                "flip_vertical": flip_vertical
            }
            
            # Apply processing
            if st.button("Process Image"):
                with st.spinner("Processing image..."):
                    # Reopen the image to ensure we're working with a fresh copy
                    image = Image.open(uploaded_file)
                    
                    # Apply processing
                    processed_image = apply_basic_processing(image, processing_options)
                    
                    # Display original and processed images side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(image, use_column_width=True)
                    
                    with col2:
                        st.markdown("**Processed Image**")
                        st.image(processed_image, use_column_width=True)
                    
                    # Option to download the processed image
                    buf = io.BytesIO()
                    processed_image.save(buf, format=image.format if image.format else "PNG")
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Processed Image",
                        data=buf,
                        file_name=f"processed_{uploaded_file.name}" if uploaded_file.name else "processed_image.png",
                        mime=f"image/{image.format.lower() if image.format else 'png'}"
                    )
                    
                    # Show histogram of processed image
                    st.markdown("#### Processed Image Histogram")
                    proc_hist_fig = get_image_histogram(processed_image)
                    st.pyplot(proc_hist_fig)
                    
                    # Show grid of transformations
                    st.markdown("#### Transformation Gallery")
                    
                    # Create individual transformations for comparison
                    transformations = []
                    titles = []
                    
                    # Original
                    transformations.append(image)
                    titles.append("Original")
                    
                    # Grayscale
                    if processing_options["grayscale"]:
                        gray_img = ImageOps.grayscale(image)
                        transformations.append(gray_img)
                        titles.append("Grayscale")
                    
                    # Brightness
                    if processing_options["brightness"] != 1.0:
                        bright_img = ImageEnhance.Brightness(image).enhance(processing_options["brightness"])
                        transformations.append(bright_img)
                        titles.append(f"Brightness: {processing_options['brightness']}")
                    
                    # Contrast
                    if processing_options["contrast"] != 1.0:
                        contrast_img = ImageEnhance.Contrast(image).enhance(processing_options["contrast"])
                        transformations.append(contrast_img)
                        titles.append(f"Contrast: {processing_options['contrast']}")
                    
                    # Blur
                    if processing_options["blur_radius"] > 0:
                        blur_img = image.filter(ImageFilter.GaussianBlur(radius=processing_options["blur_radius"]))
                        transformations.append(blur_img)
                        titles.append(f"Blur: {processing_options['blur_radius']}")
                    
                    # Edge enhance
                    if processing_options["edge_enhance"]:
                        edge_img = image.filter(ImageFilter.EDGE_ENHANCE)
                        transformations.append(edge_img)
                        titles.append("Edge Enhanced")
                    
                    # Find edges
                    if processing_options["find_edges"]:
                        edges_img = image.filter(ImageFilter.FIND_EDGES)
                        transformations.append(edges_img)
                        titles.append("Edges Detected")
                    
                    # Final processed
                    transformations.append(processed_image)
                    titles.append("All Combined")
                    
                    # Create and display grid
                    if len(transformations) > 1:
                        grid_fig = create_image_grid(transformations, titles)
                        st.pyplot(grid_fig)
        
        except Exception as e:
            st.error(f"Error during image processing: {str(e)}")

# Tab 3: Image Classification
with cv_tabs[2]:
    st.markdown("### Image Classification")
    
    if transformers_available:
        st.markdown("""
        This tab uses pre-trained models to classify images.
        Upload an image in the Image Analysis tab and then use the classifier here.
        """)
        
        if 'uploaded_file' not in locals() or uploaded_file is None:
            st.info("Please upload an image in the Image Analysis tab first.")
        else:
            try:
                from transformers import pipeline
                
                # Reload the image to ensure we're working with a fresh copy
                image = Image.open(uploaded_file)
                st.image(image, caption="Image to classify", width=300)
                
                model_options = [
                    "Default (ViT)",
                    "ResNet",
                    "EfficientNet"
                ]
                
                selected_model = st.selectbox("Select classification model", model_options)
                
                if st.button("Classify Image"):
                    with st.spinner("Classifying image..."):
                        # Map selected model to actual model ID
                        model_id = "google/vit-base-patch16-224"  # Default
                        if selected_model == "ResNet":
                            model_id = "microsoft/resnet-50"
                        elif selected_model == "EfficientNet":
                            model_id = "google/efficientnet-b7"
                        
                        # Load classification pipeline
                        classifier = pipeline("image-classification", model=model_id)
                        
                        # Classify the image
                        results = classifier(image)
                        
                        # Display results
                        st.markdown("#### Classification Results")
                        
                        # Create dataframe for results
                        results_df = pd.DataFrame(results)
                        results_df.columns = ["Label", "Confidence"]
                        results_df["Confidence"] = results_df["Confidence"].apply(lambda x: f"{x:.2%}")
                        
                        is_valid, msg, problematic = validate_dataframe_for_streamlit(results_df)
                        if not is_valid:
                            st.error(f"Cannot display DataFrame: {msg}")
                        else:
                            st.dataframe(results_df)
                            
                            # Display top result
                            top_label = results[0]["label"]
                            top_score = results[0]["score"]
                            
                            st.success(f"Top prediction: **{top_label}** with {top_score:.2%} confidence")
                            
                            # Visualization of top 5 results
                            fig, ax = plt.subplots(figsize=(10, 5))
                            
                            # Limit to top 5
                            plot_data = results[:5]
                            labels = [result["label"] for result in plot_data]
                            scores = [result["score"] for result in plot_data]
                            
                            # Create horizontal bar chart
                            bars = ax.barh([label.split(',')[0] for label in labels], scores)
                            
                            # Add percentage labels
                            for i, bar in enumerate(bars):
                                ax.text(
                                    bar.get_width() + 0.01,
                                    bar.get_y() + bar.get_height()/2,
                                    f"{scores[i]:.2%}",
                                    va='center'
                                )
                            
                            ax.set_xlim(0, 1)
                            ax.set_title("Top 5 Predictions")
                            ax.set_xlabel("Confidence")
                            
                            st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Error during image classification: {str(e)}")
                st.info("Classification requires the transformers library with appropriate dependencies. Install with: pip install transformers torch")
    else:
        st.warning("Image classification requires the transformers library. Install with: pip install transformers torch")
        
        # Show placeholder for classification
        st.markdown("""
        When you install the required libraries, you'll be able to:
        
        * Classify images using state-of-the-art models
        * See confidence scores for different categories
        * Visualize classification results
        
        Install the transformers library to enable this feature.
        """)
        
        if 'uploaded_file' in locals() and uploaded_file is not None:
            st.image(Image.open(uploaded_file), caption="Your image (waiting for classifier)", width=300)

# Tab 4: Object Detection
with cv_tabs[3]:
    st.markdown("### Object Detection")
    
    if opencv_available and transformers_available:
        st.markdown("""
        This tab allows you to detect objects in images. 
        Upload an image in the Image Analysis tab first.
        """)
        
        if 'uploaded_file' not in locals() or uploaded_file is None:
            st.info("Please upload an image in the Image Analysis tab first.")
        else:
            try:
                import cv2
                from transformers import pipeline, AutoFeatureExtractor, AutoModelForObjectDetection
                
                # Reload the image
                image = Image.open(uploaded_file)
                st.image(image, caption="Image for object detection", width=400)
                
                detector_options = [
                    "Default (DETR)",
                    "YOLOv5"
                ]
                
                selected_detector = st.selectbox("Select detection model", detector_options)
                conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
                
                if st.button("Detect Objects"):
                    with st.spinner("Detecting objects..."):
                        # Map selected model to actual model ID
                        model_id = "facebook/detr-resnet-50"  # Default
                        if selected_detector == "YOLOv5":
                            model_id = "hustvl/yolos-tiny"
                        
                        # Load object detection pipeline
                        detector = pipeline("object-detection", model=model_id)
                        
                        # Detect objects
                        results = detector(image, threshold=conf_threshold)
                        
                        # Convert image for drawing
                        img_array = np.array(image)
                        img_array = img_array[:, :, ::-1].copy()  # RGB to BGR for OpenCV
                        
                        # Draw bounding boxes
                        for result in results:
                            box = result["box"]
                            label = f"{result['label']}: {result['score']:.2f}"
                            
                            # Extract coordinates
                            xmin, ymin = int(box["xmin"]), int(box["ymin"])
                            xmax, ymax = int(box["xmax"]), int(box["ymax"])
                            
                            # Draw rectangle
                            cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            
                            # Draw label background
                            cv2.rectangle(img_array, (xmin, ymin - 20), (xmin + len(label) * 10, ymin), (0, 255, 0), -1)
                            
                            # Add label
                            cv2.putText(
                                img_array, label, (xmin, ymin - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                            )
                        
                        # Convert back to RGB for display
                        img_array = img_array[:, :, ::-1]
                        
                        # Display the result
                        st.markdown("#### Detection Results")
                        st.image(img_array, caption="Object Detection Results", use_column_width=True)
                        
                        # Display detection results as table
                        if results:
                            # Create dataframe for results
                            detection_data = []
                            for i, result in enumerate(results):
                                detection_data.append({
                                    "ID": i+1,
                                    "Object": result["label"],
                                    "Confidence": f"{result['score']:.2%}",
                                    "Position": f"({int(result['box']['xmin'])}, {int(result['box']['ymin'])}, {int(result['box']['xmax'])}, {int(result['box']['ymax'])})"
                                })
                            
                            detection_df = pd.DataFrame(detection_data)
                            is_valid, msg, problematic = validate_dataframe_for_streamlit(detection_df)
                            if not is_valid:
                                st.error(f"Cannot display DataFrame: {msg}")
                            else:
                                st.dataframe(detection_df)
                                
                            st.success(f"Detected {len(results)} objects in the image.")
                        else:
                            st.info("No objects detected with the current confidence threshold. Try lowering the threshold.")
            
            except Exception as e:
                st.error(f"Error during object detection: {str(e)}")
                st.info("Object detection requires OpenCV and the transformers library with appropriate dependencies. Install with: pip install opencv-python-headless transformers torch")
    else:
        required_libs = []
        if not opencv_available:
            required_libs.append("opencv-python-headless")
        if not transformers_available:
            required_libs.append("transformers")
        
        st.warning(f"Object detection requires additional libraries. Install with: pip install {' '.join(required_libs)}")
        
        # Show placeholder for object detection
        st.markdown("""
        When you install the required libraries, you'll be able to:
        
        * Detect multiple objects in images
        * See bounding boxes around detected objects
        * Get confidence scores for each detection
        
        Install the required libraries to enable this feature.
        """)

# Footer with info
st.markdown("---")
st.markdown("**Note:** Advanced computer vision features require additional libraries. The basic image analysis and processing features work without external dependencies.") 