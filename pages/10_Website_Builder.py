"""
ALPHA - End-to-End Machine Learning Platform
Static Website Builder Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import zipfile
import io
import uuid
import base64
from pathlib import Path
from datetime import datetime
import time

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)
from utils.config import STORAGE_DIR
from utils.huggingface import query_mistral_llm, check_hf_availability, initialize_hf_api_key

# Static website templates
WEBSITE_TEMPLATES = {
    "portfolio": {
        "name": "Portfolio Website",
        "description": "A professional portfolio to showcase your projects",
        "icon": "👨‍💻",
        "preview_image": "portfolio_template.jpg",
        "pages": ["Home", "Projects", "About", "Contact"]
    },
    "landing_page": {
        "name": "Landing Page",
        "description": "A single page website to promote your product or service",
        "icon": "🚀",
        "preview_image": "landing_template.jpg",
        "pages": ["Home"]
    },
    "blog": {
        "name": "Blog",
        "description": "A simple blog to share your thoughts",
        "icon": "✍️",
        "preview_image": "blog_template.jpg",
        "pages": ["Home", "Posts", "About"]
    },
    "documentation": {
        "name": "Documentation Site",
        "description": "Document your project or API with this template",
        "icon": "📚",
        "preview_image": "docs_template.jpg",
        "pages": ["Home", "Getting Started", "API Reference", "Examples"]
    },
    "business": {
        "name": "Business Website",
        "description": "Present your business with this professional template",
        "icon": "🏢",
        "preview_image": "business_template.jpg",
        "pages": ["Home", "Services", "About", "Contact"]
    },
    "blank": {
        "name": "Blank Template",
        "description": "Start from scratch with a blank template",
        "icon": "⬜",
        "preview_image": "blank_template.jpg",
        "pages": ["Home"]
    }
}

# CSS frameworks
CSS_FRAMEWORKS = [
    {"name": "Bootstrap 5", "value": "bootstrap5"},
    {"name": "Tailwind CSS", "value": "tailwind"},
    {"name": "Bulma", "value": "bulma"},
    {"name": "Material Design", "value": "material"},
    {"name": "Custom CSS", "value": "custom"}
]

# JS frameworks/libraries
JS_FRAMEWORKS = [
    {"name": "None (Vanilla JS)", "value": "none"},
    {"name": "jQuery", "value": "jquery"},
    {"name": "Alpine.js", "value": "alpine"},
    {"name": "Simple Animation Library", "value": "animation"}
]

# Color themes
COLOR_THEMES = [
    {"name": "Light", "primary": "#4e6bff", "bg": "#ffffff", "text": "#333333"},
    {"name": "Dark", "primary": "#6e85ff", "bg": "#121212", "text": "#f5f5f5"},
    {"name": "Blue", "primary": "#0d6efd", "bg": "#f8f9fa", "text": "#212529"},
    {"name": "Green", "primary": "#198754", "bg": "#f8f9fa", "text": "#212529"},
    {"name": "Red", "primary": "#dc3545", "bg": "#f8f9fa", "text": "#212529"},
    {"name": "Purple", "primary": "#6f42c1", "bg": "#f8f9fa", "text": "#212529"},
    {"name": "Custom", "primary": "#4e54c8", "bg": "#ffffff", "text": "#333333"}
]

# Initialize session state for website configuration
if "website_config" not in st.session_state:
    st.session_state.website_config = {
        "title": "My Website",
        "description": "A beautiful static website",
        "template": None,
        "pages": [],
        "theme": {
            "primary_color": "#4e54c8",
            "background_color": "#ffffff",
            "text_color": "#333333",
            "font": "sans-serif"
        },
        "css_framework": "bootstrap5",
        "js_framework": "none",
        "custom_css": "",
        "custom_js": "",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

if "current_page_index" not in st.session_state:
    st.session_state.current_page_index = 0

if "preview_mode" not in st.session_state:
    st.session_state.preview_mode = False

# Directory for storing website projects
WEBSITES_DIR = STORAGE_DIR / "websites"
WEBSITES_DIR.mkdir(exist_ok=True, parents=True)

# Configure the page
set_page_config(title="Static Website Builder")

# Display sidebar navigation
sidebar_navigation()

# Function to generate HTML template
def generate_html_template(config):
    """Generate HTML for the website based on the configuration."""
    css_framework = config["css_framework"]
    js_framework = config["js_framework"]
    
    # CSS framework links
    css_links = {
        "bootstrap5": '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">',
        "tailwind": '<script src="https://cdn.tailwindcss.com"></script>',
        "bulma": '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css">',
        "material": '<link href="https://unpkg.com/material-components-web@latest/dist/material-components-web.min.css" rel="stylesheet">'
    }
    
    # JS framework links
    js_links = {
        "none": '',
        "jquery": '<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>',
        "alpine": '<script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>',
        "animation": '<script src="https://cdn.jsdelivr.net/npm/animejs@3.2.1/lib/anime.min.js"></script>'
    }
    
    # Additional JS based on CSS framework
    framework_js = {
        "bootstrap5": '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>',
        "tailwind": '',
        "bulma": '',
        "material": '<script src="https://unpkg.com/material-components-web@latest/dist/material-components-web.min.js"></script>'
    }
    
    # Basic HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config["title"]}</title>
    {css_links.get(css_framework, '')}
    {js_links.get(js_framework, '')}
    {framework_js.get(css_framework, '')}
    <style>
        :root {{
            --primary-color: {config["theme"]["primary_color"]};
            --background-color: {config["theme"]["background_color"]};
            --text-color: {config["theme"]["text_color"]};
        }}
        body {{
            font-family: {config["theme"]["font"]}, sans-serif;
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        .primary-color {{
            color: var(--primary-color);
        }}
        .primary-bg {{
            background-color: var(--primary-color);
            color: white;
        }}
        {config.get("custom_css", "")}
    </style>
</head>
<body>
"""
    
    # Add navigation based on CSS framework
    if css_framework == "bootstrap5":
        html += f"""    <nav class="navbar navbar-expand-lg navbar-dark primary-bg">
        <div class="container">
            <a class="navbar-brand" href="#">{config["title"]}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
"""
        # Add navigation items for each page
        for page in config["pages"]:
            html += f"""                    <li class="nav-item">
                        <a class="nav-link" href="#{page['id']}">{page['name']}</a>
                    </li>
"""
        html += """                </ul>
            </div>
        </div>
    </nav>
"""
    elif css_framework == "bulma":
        html += f"""    <nav class="navbar has-background-primary" role="navigation" aria-label="main navigation">
        <div class="container">
            <div class="navbar-brand">
                <a class="navbar-item has-text-white" href="#">{config["title"]}</a>
                <a role="button" class="navbar-burger" aria-label="menu" aria-expanded="false">
                    <span aria-hidden="true"></span>
                    <span aria-hidden="true"></span>
                    <span aria-hidden="true"></span>
                </a>
            </div>
            <div class="navbar-menu">
                <div class="navbar-end">
"""
        # Add navigation items for each page
        for page in config["pages"]:
            html += f"""                    <a class="navbar-item has-text-white" href="#{page['id']}">{page['name']}</a>
"""
        html += """                </div>
            </div>
        </div>
    </nav>
"""
    
    # Add content for homepage
    if config["pages"] and css_framework == "bootstrap5":
        html += f"""    <header class="py-5 mb-5 primary-bg">
        <div class="container">
            <div class="row">
                <div class="col-lg-12">
                    <h1 class="display-4">{config["title"]}</h1>
                    <p class="lead">{config["description"]}</p>
                </div>
            </div>
        </div>
    </header>
    
    <main class="container my-5">
        <!-- Page content will go here -->
        <div class="row">
            <div class="col-lg-8 mb-4">
                <h2>Welcome to {config["title"]}</h2>
                <p>This is a sample page content. Edit this to add your own content.</p>
            </div>
            <div class="col-lg-4 mb-4">
                <div class="card h-100">
                    <div class="card-body">
                        <h2 class="card-title">Sidebar</h2>
                        <p class="card-text">Add a sidebar content here.</p>
                    </div>
                </div>
            </div>
        </div>
    </main>
"""
    
    # Add footer
    if css_framework == "bootstrap5":
        html += """    <footer class="py-4 primary-bg">
        <div class="container">
            <p class="m-0 text-center text-white">Copyright &copy; Your Website 2023</p>
        </div>
    </footer>
"""
    
    # Add custom JavaScript
    html += f"""    <script>
        // Add your custom JavaScript here
        {config.get("custom_js", "")}
    </script>
</body>
</html>
"""
    
    return html

# Function to create zip file with website files
def download_website_files(config):
    """Create a zip file containing all website files."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add index.html
        html_content = generate_html_template(config)
        zip_file.writestr("index.html", html_content)
        
        # Add CSS file
        css_content = f"""/* Custom CSS for {config["title"]} */
:root {{
    --primary-color: {config["theme"]["primary_color"]};
    --background-color: {config["theme"]["background_color"]};
    --text-color: {config["theme"]["text_color"]};
}}

body {{
    font-family: {config["theme"]["font"]}, sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}}

.primary-color {{
    color: var(--primary-color);
}}

.primary-bg {{
    background-color: var(--primary-color);
    color: white;
}}

/* Custom styles */
{config.get("custom_css", "")}
"""
        zip_file.writestr("css/style.css", css_content)
        
        # Add JavaScript file
        js_content = f"""// Custom JavaScript for {config["title"]}
document.addEventListener('DOMContentLoaded', function() {{
    console.log('Website loaded successfully!');
    
    // Your custom JavaScript code here
    {config.get("custom_js", "")}
}});
"""
        zip_file.writestr("js/script.js", js_content)
        
        # Add README file
        readme_content = f"""# {config["title"]}

{config["description"]}

## About This Website

This website was created using the Static Website Builder in the ML Model Builder application.

## Files Structure

- index.html - Main HTML file
- css/style.css - Custom CSS styles
- js/script.js - Custom JavaScript code

## CSS Framework

This website uses {config["css_framework"]} CSS framework.

## JavaScript

This website uses {config["js_framework"]} for JavaScript functionality.

## Customization

To customize this website further, you can edit the HTML, CSS, and JavaScript files directly.
"""
        zip_file.writestr("README.md", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer

# Function to save website configuration
def save_website_config():
    """Save the current website configuration."""
    website_id = st.session_state.website_config.get("id", str(uuid.uuid4()))
    st.session_state.website_config["id"] = website_id
    st.session_state.website_config["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    website_dir = WEBSITES_DIR / website_id
    website_dir.mkdir(exist_ok=True, parents=True)
    
    with open(website_dir / "config.json", "w") as f:
        json.dump(st.session_state.website_config, f, indent=2)
    
    # Generate and save HTML file for preview
    html_content = generate_html_template(st.session_state.website_config)
    with open(website_dir / "index.html", "w") as f:
        f.write(html_content)
    
    return website_id

# Function to load website configuration
def load_website_config(website_id):
    """Load a website configuration by ID."""
    website_dir = WEBSITES_DIR / website_id
    if not website_dir.exists():
        return None
    
    try:
        with open(website_dir / "config.json", "r") as f:
            config = json.load(f)
        
        return config
    except Exception as e:
        st.error(f"Error loading website configuration: {str(e)}")
        return None

# Function to list saved websites
def list_saved_websites():
    """List all saved website projects."""
    websites = []
    for website_dir in WEBSITES_DIR.glob("*"):
        if website_dir.is_dir():
            config_file = website_dir / "config.json"
            if config_file.exists():
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                    
                    websites.append({
                        "id": config.get("id", website_dir.name),
                        "title": config.get("title", "Untitled"),
                        "description": config.get("description", ""),
                        "created_at": config.get("created_at", ""),
                        "last_updated": config.get("last_updated", "")
                    })
                except:
                    pass
    
    return websites

# Function to set theme from preset
def set_theme_from_preset(theme_name):
    """Set theme colors based on preset name."""
    for theme in COLOR_THEMES:
        if theme["name"].lower() == theme_name.lower():
            st.session_state.website_config["theme"]["primary_color"] = theme["primary"]
            st.session_state.website_config["theme"]["background_color"] = theme["bg"]
            st.session_state.website_config["theme"]["text_color"] = theme["text"]
            break

# Main content
page_header(
    title="Static Website Builder",
    description="Build beautiful static HTML, CSS, and JavaScript websites",
    icon="🌐"
)

# Check if in preview mode
if st.session_state.preview_mode:
    # Sidebar controls for preview mode
    with st.sidebar:
        st.markdown("### Website Preview")
        if st.button("Exit Preview Mode"):
            st.session_state.preview_mode = False
            st.rerun()
    
    # Display website preview
    html_content = generate_html_template(st.session_state.website_config)
    
    st.markdown("### Website Preview")
    
    # Create a download button for the website files
    website_zip = download_website_files(st.session_state.website_config)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.download_button(
            label="Download Website Files",
            data=website_zip,
            file_name=f"{st.session_state.website_config['title'].lower().replace(' ', '_')}_website.zip",
            mime="application/zip",
            help="Download all the files for your website in a zip archive."
        )
    
    with col2:
        if st.button("Save Website Project"):
            website_id = save_website_config()
            st.success(f"Website project saved successfully!")
    
    # Display HTML preview in an iframe
    st.components.v1.html(html_content, height=600, scrolling=True)
    
    # Display HTML code with syntax highlighting
    with st.expander("View HTML Code"):
        st.code(html_content, language="html")
else:
    # Sidebar settings
    with st.sidebar:
        st.markdown("### Website Settings")
        
        if st.button("Preview Website"):
            st.session_state.preview_mode = True
            st.rerun()
        
        st.markdown("### Theme")
        theme_preset = st.selectbox(
            "Choose a color theme",
            [theme["name"] for theme in COLOR_THEMES]
        )
        if st.button("Apply Theme"):
            set_theme_from_preset(theme_preset)
    
    # Create tabs for different builder sections
    website_tabs = create_tab_panels(
        "Template Gallery", "Website Editor", "HTML/CSS/JS Editor", "AI Generation", "Manage Websites"
    )
    
    # Tab 1: Template Gallery
    with website_tabs[0]:
        st.markdown("### Choose a Template")
        
        # Display templates in a grid
        template_cols = st.columns(3)
        
        for i, (template_id, template) in enumerate(WEBSITE_TEMPLATES.items()):
            with template_cols[i % 3]:
                st.markdown(f"### {template['icon']} {template['name']}")
                st.markdown(template["description"])
                if st.button(f"Use Template", key=f"template_{template_id}"):
                    # Initialize website with template
                    st.session_state.website_config["template"] = template_id
                    st.session_state.website_config["pages"] = [
                        {"id": str(uuid.uuid4()), "name": page, "components": []}
                        for page in template["pages"]
                    ]
                    st.success(f"Template '{template['name']}' selected! Switch to the Website Editor tab.")
    
    # Tab 2: Website Editor
    with website_tabs[1]:
        st.markdown("### Website Configuration")
        
        # Basic website info
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.website_config["title"] = st.text_input(
                "Website Title",
                value=st.session_state.website_config["title"]
            )
        
        with col2:
            selected_framework = st.selectbox(
                "CSS Framework",
                options=[fw["value"] for fw in CSS_FRAMEWORKS],
                format_func=lambda x: next((fw["name"] for fw in CSS_FRAMEWORKS if fw["value"] == x), x),
                index=[fw["value"] for fw in CSS_FRAMEWORKS].index(st.session_state.website_config["css_framework"])
            )
            st.session_state.website_config["css_framework"] = selected_framework
        
        # Website description
        st.session_state.website_config["description"] = st.text_area(
            "Website Description",
            value=st.session_state.website_config["description"],
            height=100
        )
        
        # Theme customization
        st.markdown("### Theme Customization")
        theme_cols = st.columns(3)
        
        with theme_cols[0]:
            st.session_state.website_config["theme"]["primary_color"] = st.color_picker(
                "Primary Color",
                value=st.session_state.website_config["theme"]["primary_color"]
            )
        
        with theme_cols[1]:
            st.session_state.website_config["theme"]["background_color"] = st.color_picker(
                "Background Color",
                value=st.session_state.website_config["theme"]["background_color"]
            )
        
        with theme_cols[2]:
            st.session_state.website_config["theme"]["text_color"] = st.color_picker(
                "Text Color",
                value=st.session_state.website_config["theme"]["text_color"]
            )
        
        font_options = ["sans-serif", "serif", "monospace", "Arial", "Helvetica", "Georgia", "Times New Roman", "Courier New"]
        st.session_state.website_config["theme"]["font"] = st.selectbox(
            "Font Family",
            options=font_options,
            index=font_options.index(st.session_state.website_config["theme"]["font"]) if st.session_state.website_config["theme"]["font"] in font_options else 0
        )
        
        # Page management
        st.markdown("### Pages")
        
        if not st.session_state.website_config["pages"]:
            st.warning("No pages added yet. Select a template or add pages manually.")
        else:
            page_cols = st.columns(3)
            
            for i, page in enumerate(st.session_state.website_config["pages"]):
                with page_cols[i % 3]:
                    st.markdown(f"**{page['name']}**")
        
        new_page_name = st.text_input("New Page Name")
        if st.button("Add Page") and new_page_name:
            st.session_state.website_config["pages"].append({
                "id": str(uuid.uuid4()),
                "name": new_page_name,
                "components": []
            })
            st.success(f"Page '{new_page_name}' added!")
    
    # Tab 3: HTML/CSS/JS Editor
    with website_tabs[2]:
        st.markdown("### Code Editor")
        
        code_tabs = st.tabs(["HTML", "CSS", "JavaScript"])
        
        with code_tabs[0]:
            st.markdown("### HTML Editor")
            st.info("This will be added to the body of your HTML file.")
            custom_html = st.text_area(
                "Custom HTML",
                value=st.session_state.website_config.get("custom_html", ""),
                height=300
            )
            st.session_state.website_config["custom_html"] = custom_html
        
        with code_tabs[1]:
            st.markdown("### CSS Editor")
            custom_css = st.text_area(
                "Custom CSS",
                value=st.session_state.website_config.get("custom_css", ""),
                height=300
            )
            st.session_state.website_config["custom_css"] = custom_css
        
        with code_tabs[2]:
            st.markdown("### JavaScript Editor")
            custom_js = st.text_area(
                "Custom JavaScript",
                value=st.session_state.website_config.get("custom_js", ""),
                height=300
            )
            st.session_state.website_config["custom_js"] = custom_js
    
    # Tab 4: AI Generation
    with website_tabs[3]:
        st.markdown("### 🤖 AI Website Generation")
        st.markdown("Use AI to generate website content and code based on your specifications.")
        
        # Check Hugging Face API availability
        hf_status = check_hf_availability()
        
        # LLM model selection
        st.sidebar.markdown("### AI Generator Settings")
        llm_model = st.sidebar.selectbox(
            "Select LLM Model",
            options=[
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/flan-t5-base",
                "gpt2",
                "facebook/opt-350m",
                "stabilityai/stablelm-3b-4e1t",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "Local Text Generation (Rule-Based)"
            ],
            help="Choose which model to use for text generation. 'Local Text Generation' works without API keys."
        )
        
        # Simple local text generation function
        def local_text_generation(prompt, max_length=500):
            """Generate text without using external APIs - simple rule-based approach"""
            # Extract key information from the prompt
            prompt_lower = prompt.lower()
            
            # Basic HTML snippets library
            html_components = {
                "navigation": """<nav class="navbar">
  <div class="logo">Your Logo</div>
  <ul class="nav-links">
    <li><a href="#">Home</a></li>
    <li><a href="#">About</a></li>
    <li><a href="#">Services</a></li>
    <li><a href="#">Contact</a></li>
  </ul>
</nav>""",
                "hero": """<section class="hero">
  <h1>Welcome to Our Website</h1>
  <p>Your amazing tagline goes here.</p>
  <button class="cta-button">Get Started</button>
</section>""",
                "about": """<section class="about">
  <h2>About Us</h2>
  <p>We are a dedicated team of professionals committed to excellence in our field. With years of experience and a passion for what we do, we deliver results that exceed expectations.</p>
</section>""",
                "services": """<section class="services">
  <h2>Our Services</h2>
  <div class="service-cards">
    <div class="card">
      <h3>Service 1</h3>
      <p>Description of service 1 goes here.</p>
    </div>
    <div class="card">
      <h3>Service 2</h3>
      <p>Description of service 2 goes here.</p>
    </div>
    <div class="card">
      <h3>Service 3</h3>
      <p>Description of service 3 goes here.</p>
    </div>
  </div>
</section>""",
                "contact": """<section class="contact">
  <h2>Contact Us</h2>
  <form>
    <input type="text" placeholder="Your Name">
    <input type="email" placeholder="Your Email">
    <textarea placeholder="Your Message"></textarea>
    <button type="submit">Send</button>
  </form>
</section>"""
            }
            
            # CSS snippets library
            css_components = {
                "base": """body {
  font-family: 'Arial', sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 0;
  padding: 0;
}

h1, h2, h3 {
  margin-bottom: 1rem;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
}""",
                "navbar": """.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background-color: #333;
  color: white;
}

.nav-links {
  display: flex;
  list-style: none;
}

.nav-links li {
  margin-left: 20px;
}

.nav-links a {
  color: white;
  text-decoration: none;
}""",
                "hero": """.hero {
  text-align: center;
  padding: 100px 20px;
  background-color: #f5f5f5;
}

.hero h1 {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.cta-button {
  padding: 10px 20px;
  background-color: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 1rem;
}"""
            }
            
            # Check what type of content is requested
            if "html" in prompt_lower:
                # Determine what HTML components are needed
                result = "<!-- Generated HTML Code -->\n"
                
                if "nav" in prompt_lower or "menu" in prompt_lower or "header" in prompt_lower:
                    result += html_components["navigation"] + "\n\n"
                    
                if "hero" in prompt_lower or "banner" in prompt_lower or "welcome" in prompt_lower:
                    result += html_components["hero"] + "\n\n"
                    
                if "about" in prompt_lower:
                    result += html_components["about"] + "\n\n"
                    
                if "service" in prompt_lower:
                    result += html_components["services"] + "\n\n"
                    
                if "contact" in prompt_lower or "form" in prompt_lower:
                    result += html_components["contact"] + "\n\n"
                    
                # If no specific components were detected, return a basic structure
                if len(result) <= 30:
                    result += """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Your Website</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <h1>Your Website</h1>
    </header>
    <main>
        <section>
            <h2>Welcome</h2>
            <p>This is a sample website. Replace this content with your own.</p>
        </section>
    </main>
    <footer>
        <p>&copy; 2023 Your Company. All rights reserved.</p>
    </footer>
</body>
</html>"""
                
                return {"generated_text": result}
                
            elif "css" in prompt_lower:
                # Determine what CSS components are needed
                result = "/* Generated CSS Styles */\n"
                
                # Always include base styles
                result += css_components["base"] + "\n\n"
                
                if "nav" in prompt_lower or "menu" in prompt_lower or "header" in prompt_lower:
                    result += css_components["navbar"] + "\n\n"
                    
                if "hero" in prompt_lower or "banner" in prompt_lower:
                    result += css_components["hero"] + "\n\n"
                    
                # Add responsive design if mentioned
                if "responsive" in prompt_lower or "mobile" in prompt_lower:
                    result += """/* Responsive Design */
@media (max-width: 768px) {
  .navbar {
    flex-direction: column;
  }
  
  .nav-links {
    margin-top: 1rem;
  }
  
  .hero h1 {
    font-size: 2rem;
  }
}"""
                
                return {"generated_text": result}
                
            else:
                # Text content generation
                if "about" in prompt_lower:
                    return {"generated_text": """# About Our Company

We are a forward-thinking organization dedicated to innovation and excellence. With over 10 years of experience in the industry, we've built a reputation for delivering high-quality solutions that meet the unique needs of our clients.

## Our Mission

Our mission is to empower businesses through cutting-edge technology and exceptional service. We believe in creating lasting partnerships with our clients, understanding their challenges, and developing customized solutions that drive success.

## Our Team

Our team consists of passionate experts from diverse backgrounds, bringing together a wealth of knowledge and creativity. We're united by our commitment to excellence and our drive to push boundaries in everything we do."""}
                elif "services" in prompt_lower:
                    return {"generated_text": """# Our Services

## Web Development
We create beautiful, responsive websites tailored to your unique needs. Our development team uses the latest technologies to ensure your site is fast, secure, and user-friendly.

## Digital Marketing
Boost your online presence with our comprehensive digital marketing strategies. From SEO to social media management, we'll help you reach your target audience effectively.

## Consulting
Our experienced consultants provide valuable insights to help you overcome challenges and seize opportunities. We analyze your business processes and recommend practical solutions for growth."""}
                elif "contact" in prompt_lower:
                    return {"generated_text": """# Contact Us

We'd love to hear from you! Whether you have a question about our services, need a quote, or want to discuss a project, our team is ready to assist you.

## Get in Touch

- **Phone:** (555) 123-4567
- **Email:** info@yourcompany.com
- **Address:** 123 Business Street, Suite 100, City, State, ZIP

## Office Hours

Monday - Friday: 9:00 AM - 5:00 PM
Saturday - Sunday: Closed

## Send Us a Message

Fill out the form below, and we'll get back to you as soon as possible."""}
                else:
                    return {"generated_text": """# Welcome to Our Website

Thank you for visiting our website! We're dedicated to providing you with valuable information and exceptional services to meet your needs.

## What We Offer

- Professional expertise in our field
- Customized solutions tailored to your requirements
- Dedicated support from our experienced team
- Innovative approaches to solve complex challenges

Explore our website to learn more about who we are and how we can help you achieve your goals. If you have any questions or would like to discuss your specific needs, don't hesitate to reach out to us."""}
        
        # If API is not available, warn but still provide functionality
        if hf_status["status"] == "not_available":
            st.warning("⚠️ Hugging Face API connection is not available. Using local text generation instead.")
            
            with st.expander("How to set up Hugging Face API for better results"):
                st.markdown("""
                1. Go to the Hugging Face page in the sidebar
                2. Enter your API key in the sidebar
                3. Return to this page after setup
                
                Using your API key will enable access to more powerful AI models for better generation quality.
                """)
            
            st.markdown("### 📝 Generation is available in offline mode")
            st.info("While the API connection is unavailable, you can still use the local generation feature. The quality may be limited compared to online models.")
        
        # Content generation
        st.markdown("### 📝 Generate Website Content")
        
        tab1, tab2, tab3 = st.tabs(["Content Generation", "HTML Generation", "CSS Generation"])
        
        with tab1:
            website_description = st.text_area(
                "Describe your website", 
                placeholder="e.g., I need a professional portfolio website for a photographer with sections for gallery, services, about me, and contact information.",
                height=150
            )
            
            content_type = st.selectbox(
                "What type of content do you need?",
                options=["Full website text", "About page", "Services description", "Hero section", "Call-to-action", "Custom section"]
            )
            
            tone = st.selectbox(
                "Tone of voice",
                options=["Professional", "Casual", "Technical", "Creative", "Formal", "Friendly"]
            )
            
            if st.button("Generate Content", key="gen_content"):
                if not website_description:
                    st.warning("Please provide a description of your website first.")
                else:
                    with st.spinner("Generating content using AI..."):
                        prompt = f"""Generate website content for:
                        Type: {content_type}
                        Tone: {tone}
                        Website description: {website_description}
                        
                        Please provide well-structured content that can be used directly on a website. 
                        Format the content with appropriate headlines, paragraphs, and bullet points if needed.
                        """
                        
                        # Use local generation if API is unavailable or user selected the local option
                        if hf_status["status"] == "not_available" or llm_model == "Local Text Generation (Rule-Based)":
                            result = local_text_generation(prompt)
                        else:
                            # Try the selected model
                            result = query_mistral_llm(prompt, max_length=500, model_name=llm_model)
                            
                            # Fall back to local generation if API call fails
                            if "error" in result:
                                st.warning(f"API error: {result.get('error')}. Falling back to local generation.")
                                result = local_text_generation(prompt)
                        
                        st.success("Content generated successfully!")
                        generated_content = result.get("generated_text", "")
                        st.text_area("Generated Content", value=generated_content, height=300)
                        
                        # Add button to use this content in the website
                        if st.button("Use this content"):
                            # Store in the session state for later use
                            if "generated_contents" not in st.session_state:
                                st.session_state.generated_contents = []
                            
                            st.session_state.generated_contents.append({
                                "type": content_type,
                                "content": generated_content,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            st.success("Content saved! You can add it to your website in the HTML editor.")
                            st.session_state.website_config["custom_html"] = st.session_state.website_config.get("custom_html", "") + f"\n\n<!-- {content_type} -->\n<section class='content-section'>\n{generated_content}\n</section>"
        
        with tab2:
            st.markdown("### Generate HTML Code")
            
            html_description = st.text_area(
                "Describe the HTML section you need", 
                placeholder="e.g., Create a responsive navigation bar with logo and 4 menu items: Home, Services, Gallery, Contact",
                height=100
            )
            
            html_complexity = st.select_slider(
                "Complexity level",
                options=["Simple", "Moderate", "Complex"]
            )
            
            html_framework = st.selectbox(
                "HTML Framework",
                options=["None (Pure HTML)", "Bootstrap 5", "Tailwind CSS", "Semantic UI"]
            )
            
            if st.button("Generate HTML", key="gen_html"):
                if not html_description:
                    st.warning("Please provide a description of the HTML section you need.")
                else:
                    with st.spinner("Generating HTML code using AI..."):
                        prompt = f"""Generate HTML code based on the following description:
                        Description: {html_description}
                        Complexity: {html_complexity}
                        Framework: {html_framework}
                        
                        Provide only the HTML code without explanations. Make sure it's valid HTML5 and well-formatted.
                        If using Bootstrap or other frameworks, include the appropriate classes.
                        """
                        
                        if hf_status["status"] == "not_available" or llm_model == "Local Text Generation (Rule-Based)":
                            result = local_text_generation(prompt)
                        else:
                            result = query_mistral_llm(prompt, max_length=800, model_name=llm_model)
                            if "error" in result:
                                st.warning(f"API error: {result.get('error')}. Falling back to local generation.")
                                result = local_text_generation(prompt)
                        
                        st.success("HTML code generated successfully!")
                        generated_html = result.get("generated_text", "")
                        
                        # Try to clean up the code (remove markdown code blocks if present)
                        if "```html" in generated_html:
                            generated_html = generated_html.split("```html")[1].split("```")[0].strip()
                        elif "```" in generated_html:
                            generated_html = generated_html.split("```")[1].split("```")[0].strip()
                        
                        st.code(generated_html, language="html")
                        
                        # Add button to use this HTML
                        if st.button("Use this HTML"):
                            st.session_state.website_config["custom_html"] = st.session_state.website_config.get("custom_html", "") + "\n\n" + generated_html
                            st.success("HTML added to your website! You can view and edit it in the HTML/CSS/JS Editor tab.")
        
        with tab3:
            st.markdown("### Generate CSS Code")
            
            css_description = st.text_area(
                "Describe the CSS styles you need", 
                placeholder="e.g., Create a modern, responsive card layout with hover effects and subtle shadows",
                height=100
            )
            
            css_color_scheme = st.selectbox(
                "Color scheme",
                options=["Use website theme colors", "Blue/Gray professional", "Earthy/Natural tones", "Bold and vibrant", "Minimal monochrome", "Custom"]
            )
            
            if css_color_scheme == "Custom":
                col1, col2, col3 = st.columns(3)
                with col1:
                    primary_color = st.color_picker("Primary color", value="#4e54c8")
                with col2:
                    secondary_color = st.color_picker("Secondary color", value="#8f94fb")
                with col3:
                    accent_color = st.color_picker("Accent color", value="#ff5e62")
                
                custom_colors = f"Primary: {primary_color}, Secondary: {secondary_color}, Accent: {accent_color}"
            else:
                custom_colors = None
            
            if st.button("Generate CSS", key="gen_css"):
                if not css_description:
                    st.warning("Please provide a description of the CSS styles you need.")
                else:
                    with st.spinner("Generating CSS code using AI..."):
                        color_info = custom_colors if css_color_scheme == "Custom" else css_color_scheme
                        
                        prompt = f"""Generate CSS code based on the following description:
                        Description: {css_description}
                        Color scheme: {color_info}
                        
                        Provide only the CSS code without explanations. Make sure it's valid CSS3 and well-formatted.
                        Include appropriate media queries for responsiveness if applicable.
                        """
                        
                        result = query_mistral_llm(prompt, max_length=800)
                        
                        if "error" in result:
                            st.error(f"Error: {result.get('error')} - {result.get('message', '')}")
                        else:
                            st.success("CSS code generated successfully!")
                            generated_css = result.get("generated_text", "")
                            
                            # Try to clean up the code (remove markdown code blocks if present)
                            if "```css" in generated_css:
                                generated_css = generated_css.split("```css")[1].split("```")[0].strip()
                            elif "```" in generated_css:
                                generated_css = generated_css.split("```")[1].split("```")[0].strip()
                            
                            st.code(generated_css, language="css")
                            
                            # Add button to use this CSS
                            if st.button("Use this CSS"):
                                st.session_state.website_config["custom_css"] = st.session_state.website_config.get("custom_css", "") + "\n\n" + generated_css
                                st.success("CSS added to your website! You can view and edit it in the HTML/CSS/JS Editor tab.")
                                
        # History of generated content
        if "generated_contents" in st.session_state and st.session_state.generated_contents:
            with st.expander("Previously Generated Content"):
                for i, content in enumerate(st.session_state.generated_contents):
                    st.markdown(f"**{content['type']}** - {content['timestamp']}")
                    st.text_area(f"Content {i+1}", value=content['content'], height=100, key=f"prev_content_{i}")
                    st.markdown("---")
    
    # Tab 5: Manage Websites
    with website_tabs[4]:
        st.markdown("### Saved Websites")
        
        if st.button("Save Current Website"):
            website_id = save_website_config()
            st.success(f"Website saved successfully!")
        
        saved_websites = list_saved_websites()
        
        if not saved_websites:
            st.info("No saved websites yet. Create and save a website first.")
        else:
            for website in saved_websites:
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**{website['title']}**")
                    st.caption(website['description'])
                
                with col2:
                    st.caption(f"Last updated: {website['last_updated']}")
                
                with col3:
                    if st.button("Load", key=f"load_{website['id']}"):
                        loaded_config = load_website_config(website['id'])
                        if loaded_config:
                            st.session_state.website_config = loaded_config
                            st.success(f"Website '{website['title']}' loaded!")
                            st.rerun()
                
                with col4:
                    if st.button("Preview", key=f"preview_{website['id']}"):
                        loaded_config = load_website_config(website['id'])
                        if loaded_config:
                            st.session_state.website_config = loaded_config
                            st.session_state.preview_mode = True
                            st.rerun()

# Footer with information
st.markdown("---")
display_info_box(
    text="""
    **Static Website Builder Help**
    
    1. Choose a template or start from scratch
    2. Configure your website's title, description, and theme
    3. Add and manage pages
    4. Add custom HTML, CSS, or JavaScript
    5. Preview your website
    6. Download the website files
    
    All downloaded files can be directly uploaded to any web hosting service.
    """,
    type="info"
) 