"""
UI utility module for the ALPHA platform.
Contains UI helper functions and reusable components.
"""

import os
import streamlit as st
import base64
from pathlib import Path
from utils.imports import is_package_available

def load_css():
    """Load the custom CSS for styling the application."""
    # Get the directory of the current file
    current_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    css_file = current_dir / "styles" / "main.css"
    
    with open(css_file, "r") as f:
        css = f.read()
    
    # Inject CSS with the st.markdown function
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def set_page_config(title="ALPHA", layout="wide"):
    """Set the page configuration with consistent settings."""
    st.set_page_config(
        page_title=f"ALPHA - {title}",
        page_icon="🧠",
        layout=layout,
        initial_sidebar_state="expanded",
    )
    
    # Load custom CSS
    load_css()

def page_header(title, description=None, icon=None):
    """Create a consistent page header with title and optional description."""
    header_container = st.container()
    
    with header_container:
        col1, col2 = st.columns([9, 1])
        
        with col1:
            if icon:
                st.markdown(f"<h1>{icon} {title}</h1>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
            
            if description:
                st.markdown(f"<p>{description}</p>", unsafe_allow_html=True)
        
        with col2:
            # Add help button with question mark icon
            with st.expander("ℹ️"):
                st.markdown("""
                    Need help with this page?
                    Use the AI Assistant in the sidebar
                    or check the documentation.
                """)

def create_card(title, content, icon=None, color="primary"):
    """Create a styled card with title and content."""
    if color == "primary":
        card_style = "background: linear-gradient(90deg, #4e54c8, #7377de);"
    elif color == "secondary":
        card_style = "background: linear-gradient(90deg, #ff6b6b, #ff9d9d);"
    elif color == "accent":
        card_style = "background: linear-gradient(90deg, #34eba8, #7af2c5);"
    else:
        card_style = "background: #ffffff;"
    
    icon_html = f"{icon} " if icon else ""
    
    card_html = f"""
    <div style="{card_style} border-radius: 0.8rem; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);">
        <h3 style="color: white; margin-top: 0;">{icon_html}{title}</h3>
        <div style="color: white;">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_metric_card(title, value, delta=None, delta_color="normal", help_text=None):
    """Create a metric card with title, value, and optional delta."""
    col = st.container()
    
    with col:
        if help_text:
            title = f"{title} ℹ️"
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help_text
            )
        else:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )

def local_css(file_name):
    """Load local CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def remote_css(url):
    """Load remote CSS."""
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def display_info_box(text, type="info"):
    """Display a styled info box with different types: info, success, warning, error."""
    if type == "info":
        box_class = "info-box"
        icon = "ℹ️"
    elif type == "success":
        box_class = "success-box"
        icon = "✅"
    elif type == "warning":
        box_class = "warning-box"
        icon = "⚠️"
    elif type == "error":
        box_class = "error-box"
        icon = "❌"
    else:
        box_class = "info-box"
        icon = "ℹ️"
    
    st.markdown(f"""
    <div class="{box_class}">
        <p>{icon} {text}</p>
    </div>
    """, unsafe_allow_html=True)

def add_vertical_space(num_lines=1):
    """Add vertical space using HTML BR tags."""
    for _ in range(num_lines):
        st.markdown("<br>", unsafe_allow_html=True)

def gradient_text(text, gradient="primary"):
    """Display text with a gradient color."""
    if gradient == "primary":
        grad = "linear-gradient(90deg, #4e54c8, #7377de)"
    elif gradient == "secondary":
        grad = "linear-gradient(90deg, #ff6b6b, #ff9d9d)"
    elif gradient == "accent":
        grad = "linear-gradient(90deg, #34eba8, #7af2c5)"
    else:
        grad = gradient
    
    st.markdown(f"""
    <h1 style="background-image: {grad}; 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               display: inline-block;">
        {text}
    </h1>
    """, unsafe_allow_html=True)

def display_step_header(step_number, title, description=None):
    """Display a step header with step number, title, and optional description."""
    col1, col2 = st.columns([1, 11])
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #4e54c8; 
                    color: white; 
                    border-radius: 50%; 
                    width: 40px; 
                    height: 40px; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center;
                    font-weight: bold;
                    font-size: 18px;">
            {step_number}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h3>{title}</h3>", unsafe_allow_html=True)
        if description:
            st.markdown(f"<p>{description}</p>", unsafe_allow_html=True)

def display_file_download_link(file_path, link_text):
    """Generate a download link for a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
    st.markdown(href, unsafe_allow_html=True)

def display_code_block(code, language="python"):
    """Display a syntax-highlighted code block."""
    st.code(code, language=language)

def display_json_viewer(json_data):
    """Display a JSON object in a nice viewer."""
    import json
    
    # Convert to JSON string if it's an object
    if not isinstance(json_data, str):
        json_str = json.dumps(json_data, indent=2)
    else:
        json_str = json_data
    
    st.json(json.loads(json_str))

def create_tab_panels(*tab_names):
    """Create and return a set of tabs."""
    return st.tabs(tab_names)

def display_color_palette(colors, title="Color Palette"):
    """Display a color palette."""
    st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
    
    palette_html = '<div style="display: flex; gap: 10px; flex-wrap: wrap;">'
    
    for color in colors:
        palette_html += f'''
        <div style="background-color: {color}; 
                     width: 50px; 
                     height: 50px; 
                     border-radius: 5px;
                     display: flex;
                     align-items: center;
                     justify-content: center;
                     color: white;
                     font-size: 10px;
                     text-shadow: 0 0 2px black;">
            {color}
        </div>
        '''
    
    palette_html += '</div>'
    
    st.markdown(palette_html, unsafe_allow_html=True)

def display_dataframe_with_download(df, filename="data.csv"):
    """Display a dataframe with a download button."""
    import pandas as pd
    import io
    
    # Handle potential PyArrow serialization issues
    try:
        # Display the dataframe
        st.dataframe(df)
    except Exception as e:
        st.warning("Warning: Unable to display dataframe with Streamlit's native renderer. Using fallback.")
        # Fallback to HTML representation
        st.write(df.to_html(index=False), unsafe_allow_html=True)
    
    # Create download button
    buffer = io.BytesIO()
    
    # Determine file type from filename
    if filename.endswith('.csv'):
        df.to_csv(buffer, index=False)
        mime_type = "text/csv"
    elif filename.endswith('.json'):
        df.to_json(buffer, orient="records", lines=True)
        mime_type = "application/json"
    elif filename.endswith('.xlsx'):
        if is_package_available('openpyxl'):
            df.to_excel(buffer, index=False, engine='openpyxl')
        else:
            df.to_excel(buffer, index=False)
        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        df.to_csv(buffer, index=False)
        mime_type = "text/csv"
    
    buffer.seek(0)
    st.download_button(
        label="Download data",
        data=buffer,
        file_name=filename,
        mime=mime_type,
    )

def sidebar_navigation():
    """Create sidebar navigation with custom styling."""
    st.sidebar.markdown("## 🧭 Navigation")
    
    st.sidebar.markdown("""
    <ul style="list-style-type: none; padding-left: 1rem;">
        <li><a href="/" target="_self" style="text-decoration: none; color: #4e54c8;">🏠 Home</a></li>
        <li><a href="/1_Data_Management" target="_self" style="text-decoration: none; color: #4e54c8;">📊 Data Management</a></li>
        <li><a href="/2_Model_Training" target="_self" style="text-decoration: none; color: #4e54c8;">🔬 Model Training</a></li>
        <li><a href="/5_Dashboard" target="_self" style="text-decoration: none; color: #4e54c8;">📈 Dashboard</a></li>
        <li><a href="/6_Settings" target="_self" style="text-decoration: none; color: #4e54c8;">⚙️ Settings</a></li>
        <li><a href="/7_NL_Analysis" target="_self" style="text-decoration: none; color: #4e54c8;">💬 NL Analysis</a></li>
        <li><a href="/8_ML_Assistant" target="_self" style="text-decoration: none; color: #4e54c8;">🤖 ML Assistant</a></li>
        <li><a href="/9_Computer_Vision" target="_self" style="text-decoration: none; color: #4e54c8;">👁️ Computer Vision</a></li>
        <li><a href="/10_Website_Builder" target="_self" style="text-decoration: none; color: #4e54c8;">🌐 Website Builder</a></li>
        <li><a href="/11_Hugging_Face" target="_self" style="text-decoration: none; color: #4e54c8;">🤗 Hugging Face</a></li>
        <li><a href="/12_Kaggle_Datasets" target="_self" style="text-decoration: none; color: #4e54c8;">📊 Kaggle Datasets</a></li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")

def display_notification(message, type="info"):
    """Display a notification message that can be dismissed."""
    notification_placeholder = st.empty()
    
    if type == "info":
        container = notification_placeholder.info(message)
    elif type == "success":
        container = notification_placeholder.success(message)
    elif type == "warning":
        container = notification_placeholder.warning(message)
    elif type == "error":
        container = notification_placeholder.error(message)
    else:
        container = notification_placeholder.info(message)
    
    # Auto-dismiss after 5 seconds
    import time
    time.sleep(5)
    notification_placeholder.empty()

def create_feature_list(features, columns=3):
    """Display a feature list in multiple columns with icons."""
    if not features:
        return
    
    feature_html = f'<div style="display: grid; grid-template-columns: repeat({columns}, 1fr); gap: 10px;">'
    
    for feature in features:
        icon = feature.get("icon", "✓")
        text = feature.get("text", "")
        feature_html += f'''
        <div style="display: flex; align-items: center; gap: 5px;">
            <span style="color: #4e54c8; font-weight: bold;">{icon}</span>
            <span>{text}</span>
        </div>
        '''
    
    feature_html += '</div>'
    
    st.markdown(feature_html, unsafe_allow_html=True)

def welcome_header():
    """Display a welcome header for the home page."""
    st.markdown("""
    <div style="background: linear-gradient(90deg, #4e54c8, #7377de); 
                padding: 1.5rem; 
                border-radius: 0.5rem; 
                margin-bottom: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h1 style="color: white; margin-top: 0;">Welcome to ALPHA 🧠</h1>
        <p style="color: white; font-size: 1.2rem;">
            Your end-to-end Machine Learning platform, built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display a footer with links and information."""
    from datetime import datetime
    current_year = datetime.now().year
    
    st.markdown("---")
    st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #6c757d;">
        <div>© {current_year} ALPHA ML Platform</div>
        <div>
            <a href="#" style="text-decoration: none; color: #4e54c8; margin-right: 10px;">Documentation</a>
            <a href="#" style="text-decoration: none; color: #4e54c8; margin-right: 10px;">GitHub</a>
            <a href="#" style="text-decoration: none; color: #4e54c8;">Contact</a>
        </div>
    </div>
    """, unsafe_allow_html=True) 