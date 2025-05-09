"""
ALPHA - Advanced Frontend Website Builder (HTML/CSS/JS)
A robust, user-friendly, and modern static website UI builder for rapid prototyping.
"""

# Add proper path handling for imports
import os
import sys
# Add the project root directory to the Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import json
import zipfile
import io
from pathlib import Path
import shutil
import requests
import traceback
import uuid
from dotenv import load_dotenv
import base64
from utils.helpers import load_css
from utils.imports import is_package_available
from utils.website_builder import (
    COMPONENTS, 
    css_frameworks, 
    color_schemes, 
    website_templates,
    render_html_preview, 
    generate_website_zip
)

# Load environment variables
load_dotenv()

# --- Global icon options and labels (for all components) ---
icon_options = ["", "fa-solid fa-star", "fa-solid fa-user", "fa-solid fa-heart", "fa-solid fa-rocket", "fa-solid fa-cog", "fa-solid fa-envelope"]
icon_labels = {
    "": "No Icon",
    "fa-solid fa-star": "Star",
    "fa-solid fa-user": "User",
    "fa-solid fa-heart": "Heart",
    "fa-solid fa-rocket": "Rocket",
    "fa-solid fa-cog": "Cog",
    "fa-solid fa-envelope": "Envelope"
}

# --- Global image directory for component images ---
# Ensure the directory is writable, with fallback to system temp dir
try:
    image_dir = Path("storage/websites/images")
    image_dir.mkdir(parents=True, exist_ok=True)
    # Test write permissions with a temp file
    test_file = image_dir / ".write_test"
    test_file.write_text("test")
    test_file.unlink()
except (PermissionError, OSError):
    import tempfile
    image_dir = Path(tempfile.gettempdir()) / "alpha_website_builder_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    st.warning(f"Could not write to primary image directory. Using temp directory: {image_dir}")

# --- Website Templates ---
TEMPLATES = [
    {
        "name": "Portfolio",
        "description": "A beautiful personal portfolio template.",
        "preview": "https://via.placeholder.com/800x400/2563eb/FFFFFF?text=Portfolio+Template",
        "pages": [
            {"title": "Home", "components": ["Hero", "About", "Projects", "Contact"]}
        ],
        "theme": {"primary": "#2563eb", "background": "#fff", "text": "#222"}
    },
    {
        "name": "Landing Page",
        "description": "A modern landing page for products or startups.",
        "preview": "https://via.placeholder.com/800x400/6366f1/FFFFFF?text=Landing+Page+Template",
        "pages": [
            {"title": "Home", "components": ["Hero", "Features", "CTA", "Footer"]}
        ],
        "theme": {"primary": "#6366f1", "background": "#f3f4f6", "text": "#18181b"}
    },
    {
        "name": "Blog",
        "description": "A clean blog template for sharing articles.",
        "preview": "https://via.placeholder.com/800x400/222222/FFFFFF?text=Blog+Template",
        "pages": [
            {"title": "Home", "components": ["Hero", "Posts", "Footer"]},
            {"title": "Post", "components": ["PostContent", "Comments"]}
        ],
        "theme": {"primary": "#222", "background": "#fff", "text": "#222"}
    },
    {
        "name": "Dashboard",
        "description": "A dashboard template for data visualization.",
        "preview": "https://via.placeholder.com/800x400/6366f1/FFFFFF?text=Dashboard+Template",
        "pages": [
            {"title": "Dashboard", "components": ["Header", "Charts", "Stats", "Footer"]}
        ],
        "theme": {"primary": "#6366f1", "background": "#18181b", "text": "#fff"}
    },
    {
        "name": "Blank",
        "description": "Start from scratch with a blank canvas.",
        "preview": "https://via.placeholder.com/800x400/2563eb/FFFFFF?text=Blank+Template",
        "pages": [
            {"title": "Home", "components": []}
        ],
        "theme": {"primary": "#2563eb", "background": "#fff", "text": "#222"}
    }
]

# --- Component Library ---
# Updated with more granular components and basic layouts
COMPONENTS = {
    # Content Components
    "Hero": "A large hero section with title, subtitle, and call-to-action.",
    "Text Block": "A block of paragraph text.",
    "About": "About section for personal or company info.",
    "Features": "Feature highlights for products/services.",
    "Projects": "Showcase projects or portfolio items (Card layout often used).",
    "Image Gallery": "Display multiple images in a grid.",
    "Stats": "Key statistics or KPIs.",
    "PostContent": "Content area for blog posts or articles.",
    "Comments": "Comments section for blog posts.",
    
    # Interactive Components
    "Button": "A clickable button.",
    "Contact Form": "Basic contact form structure (placeholders).",
    "Form Input": "Placeholder for a single form input field.",
    "CTA": "Call-to-action section, often with a button.",
    "Posts": "List of blog posts (links or summaries).",

    # Structural Components
    "Navbar": "Navigation bar, typically at the top.",
    "Header": "Page or section header.",
    "Footer": "Footer with links and copyright.",
    "Card": "Content container, often used in grids (e.g., for Projects).",
    "Divider": "A horizontal line to separate content.",
    "Charts": "Placeholder for data visualization charts.",

    # Layout Components (Conceptual)
    "Columns (2)": "A section divided into two columns.",
    "Columns (3)": "A section divided into three columns.",
    "Grid": "A section with items arranged in a grid.",
}

# --- Session State Initialization ---
def init_state():
    if "wb_config" not in st.session_state:
        st.session_state.wb_config = {
            "title": "My Website",
            "description": "A modern static website built with ALPHA.",
            "template": None,
            "theme": {"primary": "#2563eb", "background": "#fff", "text": "#222"},
            "pages": [
                {"title": "Home", "components": []}
            ],
        }
    if "wb_current_page" not in st.session_state:
        st.session_state.wb_current_page = 0
    if "wb_mode" not in st.session_state:
        st.session_state.wb_mode = "edit"  # or 'preview'
    if "wb_code" not in st.session_state:
        st.session_state.wb_code = {"html": "", "css": "", "js": ""}

init_state()

# --- Helper function to normalize page structure ---
def normalize_page_structure(config):
    """
    Normalize page structure to ensure compatibility between 
    UI page structure (title/components) and saved configs (id/name/components)
    """
    if not config or "pages" not in config:
        return config
        
    normalized_pages = []
    
    # Normalize theme keys and colors
    if "theme" in config:
        theme = config["theme"]
        normalized_theme = {}
        
        # Handle different key formats: primary/primary_color, etc.
        if "primary" in theme:
            normalized_theme["primary"] = theme["primary"]
        elif "primary_color" in theme:
            normalized_theme["primary"] = theme["primary_color"]
            
        if "background" in theme:
            normalized_theme["background"] = theme["background"]
        elif "background_color" in theme:
            normalized_theme["background"] = theme["background_color"]
            
        if "text" in theme:
            normalized_theme["text"] = theme["text"]
        elif "text_color" in theme:
            normalized_theme["text"] = theme["text_color"]
            
        # Ensure all values are set with fallbacks
        if "primary" not in normalized_theme:
            normalized_theme["primary"] = "#2563eb"  # Default blue
        if "background" not in normalized_theme:
            normalized_theme["background"] = "#ffffff"  # Default white
        if "text" not in normalized_theme:
            normalized_theme["text"] = "#222222"  # Default dark
            
        # Update the config with normalized theme
        config["theme"] = normalized_theme
    
    for page in config["pages"]:
        normalized_page = {}
        
        # Ensure page has an ID
        if "id" not in page:
            normalized_page["id"] = str(uuid.uuid4())
        else:
            normalized_page["id"] = page["id"]
            
        # Handle name/title field
        if "title" in page and "name" not in page:
            normalized_page["name"] = page["title"]
            normalized_page["title"] = page["title"]
        elif "name" in page and "title" not in page:
            normalized_page["title"] = page["name"]
            normalized_page["name"] = page["name"]
        elif "title" in page and "name" in page:
            # Keep both if they exist
            normalized_page["title"] = page["title"]
            normalized_page["name"] = page["name"]
        else:
            # Fallback for invalid page
            normalized_page["title"] = "Untitled Page"
            normalized_page["name"] = "Untitled Page"
            
        # Normalize components
        if "components" in page:
            normalized_page["components"] = page["components"]
        else:
            normalized_page["components"] = []
            
        normalized_pages.append(normalized_page)
    
    config["pages"] = normalized_pages
    return config

# --- Apply normalization to wb_config when loading from disk ---
if "wb_config" in st.session_state:
    st.session_state.wb_config = normalize_page_structure(st.session_state.wb_config)

# --- Sidebar Navigation ---
SECTIONS = [
    "Template",
    "Theme",
    "Pages",
    "Components",
    "Advanced Code",
    "AI Assistant",
    "Preview",
    "Export/Deploy",
    "Project Management",
    "SEO & Metadata"
]
selected_section = st.sidebar.radio("Navigation", SECTIONS, index=0)

# --- Ensure pages, selected_page_idx, and selected_page are always defined at the top ---
pages = st.session_state.wb_config["pages"]
selected_page_idx = st.session_state.get("wb_current_page", 0)
if selected_page_idx >= len(pages):
    selected_page_idx = 0
selected_page = pages[selected_page_idx]

# --- Move generate_website_zip to top level (after imports and globals) ---
def generate_website_zip(config, code):
    # Ensure css_frameworks is available
    global css_frameworks
    if 'css_frameworks' not in globals():
        css_frameworks = {
            "None": "",
            "Bootstrap 5": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
            "Tailwind CSS": "https://cdn.jsdelivr.net/npm/tailwindcss@3.3.3/dist/tailwind.min.css"
        }
        
    def render_component_export(comp_data):
        name = comp_data["name"] if isinstance(comp_data, dict) else comp_data
        img_src = comp_data.get("image", "") if isinstance(comp_data, dict) else ""
        icon = comp_data.get("icon", "") if isinstance(comp_data, dict) else ""
        props = comp_data.get("props", {}) if isinstance(comp_data, dict) else {}
        
        # Handle image and check if it exists
        img_tag = ""
        if img_src and Path(img_src).exists():
            img_path = Path(img_src)
            img_tag = f'<img src="images/{img_path.name}" alt="{name} image">'
        
        icon_html = f'<i class="{icon}"></i>' if icon else ""
        
        # Custom code override
        custom_code = props.get("custom_code", {})
        if custom_code.get("html"):
            custom_html = custom_code["html"]
            custom_css = custom_code.get("css", "")
            custom_js = custom_code.get("js", "")
            return f'<div class="component {name.lower().replace(" ", "-")}">{custom_html}<style>{custom_css}</style><script>{custom_js}</script></div>'
        if name == "Button":
            btn_text = props.get("text", "Button")
            return f'<button class="btn">{icon_html}{btn_text}</button>'
        elif name == "Card":
            card_title = props.get("title", "Card Title")
            card_content = props.get("content", "Card content...")
            return f'<div class="card">{icon_html}{img_tag}<h4>{card_title}</h4><p>{card_content}</p></div>'
        elif name == "Divider":
             return '<hr>'
        elif name == "Text Block":
             text_content = props.get("content", "Some text content...")
             return f'<div class="text-block">{icon_html}{img_tag}<p>{text_content}</p></div>'
        elif name == "Columns (2)":
             return '<div class="columns-2"><div class="column">Column 1 Content</div><div class="column">Column 2 Content</div></div>'
        elif name == "Columns (3)":
             return '<div class="columns-3"><div class="column">Column 1 Content</div><div class="column">Column 2 Content</div><div class="column">Column 3 Content</div></div>'
        elif name == "Grid":
             return '<div class="grid"><div class="grid-item">Item 1</div><div class="grid-item">Item 2</div><div class="grid-item">Item 3</div></div>'
        else:
             return f'<div class="component {name.lower().replace(" ", "-")}">{icon_html}{img_tag}<h2>{name}</h2></div>'

    try:
        page_html = []
        for p in config['pages']:
            comps = ''.join([render_component_export(c) for c in p['components']])
            page_html.append(f"<section id='{p['title'].lower().replace(' ', '-')}'><h2>{p['title']}</h2>{comps}</section>")
        # Basic CSS for structure
        base_css = f"""
        body {{ background:{config['theme']['background']}; color:{config['theme']['text']}; font-family: sans-serif; padding: 20px; }}
        h1, h2 {{ color: {config['theme']['primary']}; }}
        .btn {{ padding: 10px 15px; background-color:{config['theme']['primary']}; color:white; border:none; border-radius:5px; cursor:pointer; }}
        .card {{ border: 1px solid #eee; padding: 15px; margin-bottom: 15px; border-radius: 5px; }}
        .card img {{ max-width: 100%; height: auto; margin-bottom: 10px; }}
        hr {{ border: 0; height: 1px; background: #ccc; margin: 20px 0; }}
        .columns-2 {{ display: flex; gap: 20px; }}
        .columns-2 .column {{ flex: 1; }}
        .columns-3 {{ display: flex; gap: 20px; }}
        .columns-3 .column {{ flex: 1; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .grid-item {{ border: 1px solid #eee; padding: 10px; }}
        section {{ margin-bottom: 30px; }}
        """
        # Inject CSS framework CDN if selected
        framework_cdn = css_frameworks.get(config.get("css_framework", "None"), "")
        framework_link = f'<link rel="stylesheet" href="{framework_cdn}">' if framework_cdn else ""
        # Inject FontAwesome CDN if any icon is used
        any_icon = any(
            any(isinstance(c, dict) and c.get("icon") for c in page["components"]) for page in config["pages"]
        )
        fa_link = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">' if any_icon else ""
        meta = config.get("metadata", {})
        meta_title = meta.get("site_title", config["title"])
        meta_desc = meta.get("description", config["description"])
        favicon = meta.get("favicon", "")
        favicon_link = f'<link rel="icon" href="{favicon}">' if favicon else ""
        html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>{meta_title}</title>
    <meta name='description' content='{meta_desc}'>
    {favicon_link}
    {framework_link}
    {fa_link}
    <style>{base_css}{code['css']}</style>
</head>
<body style='background:{config['theme']['background']};color:{config['theme']['text']}'>
    <h1 style='color:{config['theme']['primary']}'>{config['title']}</h1>
    <p>{config['description']}</p>
    <!-- Pages and Components -->
    {''.join(page_html)}
    {code['html']}
    <script>{code['js']}</script>
</body>
</html>"""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("index.html", html)
            zf.writestr("styles.css", base_css + code["css"]) # Combine base and custom css
            if code["js"]:
                zf.writestr("script.js", code["js"])
            # Add images
            img_dir_export = Path("images")
            for page in config["pages"]:
                for c in page["components"]:
                    if isinstance(c, dict) and c.get("image") and Path(c["image"]).exists():
                        img_path = Path(c["image"])
                        zf.write(img_path, arcname=img_dir_export / img_path.name)
            zf.writestr("README.txt", f"Website exported from ALPHA Website Builder.\nTitle: {config['title']}\n")
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"Error creating website ZIP: {str(e)}")
        # Return a simple error page
        error_buffer = io.BytesIO()
        with zipfile.ZipFile(error_buffer, "w") as zf:
            error_html = f"""<!DOCTYPE html>
<html>
<head><title>Error</title></head>
<body>
    <h1>Error Building Website</h1>
    <p>There was an error building your website: {str(e)}</p>
    <p>Please try again or contact support.</p>
</body>
</html>"""
            zf.writestr("error.html", error_html)
        error_buffer.seek(0)
        return error_buffer

# --- Live HTML Preview Function ---
def render_html_preview(config, code, width):
    # Ensure css_frameworks is available
    global css_frameworks
    if 'css_frameworks' not in globals():
        css_frameworks = {
            "None": "",
            "Bootstrap 5": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
            "Tailwind CSS": "https://cdn.jsdelivr.net/npm/tailwindcss@3.3.3/dist/tailwind.min.css"
        }
    
    def render_component_preview(comp_data):
        name = comp_data["name"] if isinstance(comp_data, dict) else comp_data
        img_src = comp_data.get("image", "") if isinstance(comp_data, dict) else ""
        icon = comp_data.get("icon", "") if isinstance(comp_data, dict) else ""
        props = comp_data.get("props", {}) if isinstance(comp_data, dict) else {}
        icon_html = f'<i class="{icon}" style="margin-right:8px;"></i>' if icon else ""
        img_tag = f'<img src="images/{Path(img_src).name}" style="max-width:100px;max-height:60px;margin-right:8px;vertical-align:middle;">' if img_src and Path(img_src).exists() else ""
        # Custom code override
        custom_code = props.get("custom_code", {})
        if custom_code.get("html"):
            custom_html = custom_code["html"]
            custom_css = custom_code.get("css", "")
            custom_js = custom_code.get("js", "")
            return f'<div class="component-preview" style="border:1px solid #aaa; margin-bottom:10px;">{custom_html}<style>{custom_css}</style><script>{custom_js}</script></div>'
        style = "border:1px dashed #ccc; padding:10px; margin-bottom:10px; background-color:#f9f9f9; display:flex; align-items:center;"
        content = f'{icon_html}{img_tag}<span style="vertical-align:middle; font-weight:bold;">{name}</span>'
        if name == "Button":
            btn_text = props.get("text", "Button")
            content = f'<button style="padding: 5px 10px; background-color:{config["theme"]["primary"]}; color:white; border:none; border-radius:3px;">{icon_html}{btn_text}</button>'
        elif name == "Card":
            card_title = props.get("title", "Card Title")
            card_content = props.get("content", "Card content...")
            content = f'{icon_html}{img_tag}<div><h4>{card_title}</h4><p>{card_content}</p></div>'
        elif name == "Divider":
            return '<hr style="margin:15px 0;">'
        elif name == "Text Block":
            text_content = props.get("content", "This is a block of text (Lorem ipsum...)")
            content = f'<p>{icon_html}{text_content} {img_tag}</p>'
        elif name == "Columns (2)":
            content = '<div style="display:flex;gap:10px;"><div style="flex:1;border:1px dashed blue;min-height:50px;text-align:center;">Col 1</div><div style="flex:1;border:1px dashed blue;min-height:50px;text-align:center;">Col 2</div></div>'
        elif name == "Columns (3)":
            content = '<div style="display:flex;gap:10px;"><div style="flex:1;border:1px dashed green;min-height:50px;text-align:center;">Col 1</div><div style="flex:1;border:1px dashed green;min-height:50px;text-align:center;">Col 2</div><div style="flex:1;border:1px dashed green;min-height:50px;text-align:center;">Col 3</div></div>'
        elif name == "Grid":
            content = '<div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(100px, 1fr));gap:10px;"><div style="border:1px dashed orange;min-height:50px;text-align:center;">Grid Item</div><div style="border:1px dashed orange;min-height:50px;text-align:center;">Grid Item</div><div style="border:1px dashed orange;min-height:50px;text-align:center;">Grid Item</div></div>'
        return f'<div class="component-preview" style="{style}">{content}</div>'

    theme_style = f"""
    body {{ 
        background: {config['theme']['background']};
        color: {config['theme']['text']};
        font-family: sans-serif;
        padding: 20px;
    }}
    h1, h2 {{ color: {config['theme']['primary']}; }}
    section {{ margin-bottom: 20px; border-top: 1px solid #eee; padding-top:20px; }}
    .component {{ background-color: #f9f9f9; border-left: 3px solid {config['theme']['primary']}; }}
    """

    # Inject CSS framework CDN if selected
    framework_cdn = css_frameworks.get(config.get("css_framework", "None"), "")
    framework_link = f'<link rel="stylesheet" href="{framework_cdn}">' if framework_cdn else ""

    # Inject FontAwesome CDN if any icon is used
    any_icon = any(
        any(isinstance(c, dict) and c.get("icon") for c in p["components"]) for p in config["pages"]
    )
    fa_link = '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">' if any_icon else ""

    meta = config.get("metadata", {})
    meta_title = meta.get("site_title", config["title"])
    meta_desc = meta.get("description", config["description"])
    favicon = meta.get("favicon", "")
    favicon_link = f'<link rel="icon" href="{favicon}">' if favicon else ""

    html_content = (
        f"<h1 style='color:{config['theme']['primary']}'>{config['title']}</h1>"
        f"<p>{config['description']}</p>"
        + ''.join([
            f"<section><h2>{p['title']}</h2>" + ''.join([render_component_preview(c) for c in p['components']]) + "</section>"
            for p in config['pages']
        ])
        + code['html'] 
    )
    
    full_html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
    <meta charset='UTF-8'>
    <meta name='viewport' content='width=device-width, initial-scale=1.0'>
    <title>{meta_title}</title>
    <meta name='description' content='{meta_desc}'>
    {favicon_link}
    {framework_link}
    {fa_link}
    <style>{theme_style}{code['css']}</style>
</head>
<body>
    {html_content}
    <script>{code['js']}</script>
</body>
</html>"""
    st.components.v1.html(full_html, height=600, scrolling=True, width=width if isinstance(width, int) else None)

# --- Modular utility functions ---
def edit_page_title(page, idx):
    """Edit the title of a page with validation."""
    return st.text_input(f"Page Title {idx+1}", value=page["title"], key=f"edit_page_title_{idx}", help="Edit the title of this page.")

def reorder_list(items, idx, direction):
    """Move an item in a list up or down."""
    if direction == "up" and idx > 0:
        items[idx-1], items[idx] = items[idx], items[idx-1]
    elif direction == "down" and idx < len(items)-1:
        items[idx+1], items[idx] = items[idx], items[idx+1]
    return items

def display_page_list(pages):
    st.markdown("**Pages:**")
    for i, page in enumerate(pages):
        col1, col2, col3, col4 = st.columns([6, 1, 1, 2])
        with col1:
            st.write(f"{i+1}. {page['title']}")
        with col2:
            if st.button("⬆️", key=f"move_page_up_{i}", help="Move page up"):
                reorder_list(pages, i, "up")
                st.rerun()
        with col3:
            if st.button("⬇️", key=f"move_page_down_{i}", help="Move page down"):
                reorder_list(pages, i, "down")
                st.rerun()
        with col4:
            if st.button("❌", key=f"del_page_{i}", help="Delete this page"):
                pages.pop(i)
                st.rerun()

# --- AI Assistant: Prompt history ---
if "ai_prompt_history" not in st.session_state:
    st.session_state.ai_prompt_history = []
if "ai_result_history" not in st.session_state:
    st.session_state.ai_result_history = []

def add_ai_history(prompt, result):
    st.session_state.ai_prompt_history = ([prompt] + st.session_state.ai_prompt_history)[:3]
    st.session_state.ai_result_history = ([result] + st.session_state.ai_result_history)[:3]

# --- Modular utility functions for components ---
def edit_component_properties(comp, idx):
    """Edit all properties of a component, including images, icons, and custom code."""
    comp_name = comp["name"]
    comp_img = comp.get("image", "")
    comp_icon = comp.get("icon", "")
    comp_props = comp.get("props", {})
    st.write(f"{idx+1}. {comp_name} - {COMPONENTS.get(comp_name, 'Custom component')}")
    # Icon picker
    icon_val = st.selectbox(f"Icon for {comp_name}", icon_options, format_func=lambda x: icon_labels[x], key=f"icon_{idx}", index=icon_options.index(comp_icon) if comp_icon in icon_options else 0, help="Select an icon for this component.")
    comp["icon"] = icon_val
    # Image upload
    uploaded_img = st.file_uploader(f"Image for {comp_name} (optional)", type=["png", "jpg", "jpeg", "gif"], key=f"img_{idx}", help="Upload an image for this component.")
    if uploaded_img:
        try:
            img_path = image_dir / f"{st.session_state.wb_config['title'].replace(' ','_').lower()}_{idx}_{uploaded_img.name}"
            with open(img_path, "wb") as f:
                f.write(uploaded_img.getbuffer())
            comp["image"] = str(img_path)
            st.success(f"Image uploaded for {comp_name}")
        except Exception as e:
            st.error(f"Failed to save image: {str(e)}")
            st.info("Try using a different image or filename")
    if comp_img:
        try:
            if Path(comp_img).exists():
                st.image(comp_img, width=80)
            else:
                st.warning(f"Image not found: {comp_img}")
                comp["image"] = ""  # Clear reference to missing image
        except Exception:
            st.warning("Unable to display image preview")
    # Property editing (WYSIWYG)
    for prop_key, prop_val in comp_props.items():
        if prop_key not in ["custom_code"]:
            new_val = st.text_input(f"{comp_name} - {prop_key}", value=prop_val, key=f"{comp_name}_{idx}_{prop_key}", help=f"Edit the {prop_key} property.")
            comp["props"][prop_key] = new_val
    # Add default fields for common types
    if comp_name == "Button":
        btn_text = st.text_input(f"Button Text", value=comp_props.get("text", "Button"), key=f"btn_text_{idx}", help="Edit the button text.")
        comp["props"]["text"] = btn_text
    elif comp_name == "Card":
        card_title = st.text_input(f"Card Title", value=comp_props.get("title", "Card Title"), key=f"card_title_{idx}", help="Edit the card title.")
        card_content = st.text_area(f"Card Content", value=comp_props.get("content", "Card content..."), key=f"card_content_{idx}", help="Edit the card content.")
        comp["props"]["title"] = card_title
        comp["props"]["content"] = card_content
    elif comp_name == "Text Block":
        text_content = st.text_area(f"Text Content", value=comp_props.get("content", "This is a block of text (Lorem ipsum...)"), key=f"textblock_content_{idx}", help="Edit the text block content.")
        comp["props"]["content"] = text_content
    # Custom code per component
    with st.expander(f"Custom Code for {comp_name}"):
        custom_html = st.text_area(f"Custom HTML (overrides default)", value=comp_props.get("custom_code", {}).get("html", ""), key=f"custom_html_{idx}", help="Custom HTML for this component.")
        custom_css = st.text_area(f"Custom CSS", value=comp_props.get("custom_code", {}).get("css", ""), key=f"custom_css_{idx}", help="Custom CSS for this component.")
        custom_js = st.text_area(f"Custom JS", value=comp_props.get("custom_code", {}).get("js", ""), key=f"custom_js_{idx}", help="Custom JS for this component.")
        comp["props"]["custom_code"] = {"html": custom_html, "css": custom_css, "js": custom_js}

def display_component_list(components):
    for i, comp in enumerate(components):
        col1, col2, col3, col4, col5, col6 = st.columns([5, 1, 1, 1, 2, 2])
        with col1:
            edit_component_properties(comp, i)
        with col2:
            if st.button("⬆️", key=f"move_comp_up_{i}", help="Move component up") and i > 0:
                reorder_list(components, i, "up")
                st.rerun()
        with col3:
            if st.button("⬇️", key=f"move_comp_down_{i}", help="Move component down") and i < len(components)-1:
                reorder_list(components, i, "down")
                st.rerun()
        with col4:
            if st.button("❌", key=f"del_comp_{i}", help="Delete this component"):
                components.pop(i)
                st.rerun()

# --- Remove sidebar navigation and show all sections in order ---

# 1. Template
st.subheader("1️⃣ Choose a Template")
template_names = [tpl["name"] for tpl in TEMPLATES]
selected_template = st.selectbox("Select a template to start:", template_names)
template = next(t for t in TEMPLATES if t["name"] == selected_template)
st.image(template["preview"], width=400, caption=template["description"])
if st.button("Use this template", key="use_template") or st.session_state.wb_config["template"] != selected_template:
    st.session_state.wb_config = {
        "title": template["name"],
        "description": template["description"],
        "template": selected_template,
        "theme": template["theme"].copy(),
        "pages": json.loads(json.dumps(template["pages"])),
    }
    st.session_state.wb_current_page = 0
    st.success(f"Template '{selected_template}' loaded!")

# 2. Theme
st.subheader("2️⃣ Customize Theme")
css_frameworks = {
    "None": "",
    "Bootstrap 5": "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css",
    "Tailwind CSS": "https://cdn.jsdelivr.net/npm/tailwindcss@3.3.3/dist/tailwind.min.css"
}
if "css_framework" not in st.session_state.wb_config:
    st.session_state.wb_config["css_framework"] = "None"
selected_framework = st.selectbox("CSS Framework:", list(css_frameworks.keys()), index=list(css_frameworks.keys()).index(st.session_state.wb_config["css_framework"]))
st.session_state.wb_config["css_framework"] = selected_framework
framework_cdn = css_frameworks[selected_framework]
theme = st.session_state.wb_config["theme"]
col1, col2, col3 = st.columns(3)
with col1:
    theme["primary"] = st.color_picker("Primary Color", value=theme["primary"])
with col2:
    theme["background"] = st.color_picker("Background Color", value=theme["background"])
with col3:
    theme["text"] = st.color_picker("Text Color", value=theme["text"])
st.session_state.wb_config["theme"] = theme

# 3. Pages
st.subheader("3️⃣ Manage Pages")
page_titles = [p["title"] for p in pages]
selected_page_idx = st.selectbox("Select page:", range(len(pages)), format_func=lambda i: page_titles[i], key="wb_page_select")
selected_page = pages[selected_page_idx]
new_page_title = st.text_input("Add new page:", "", key="wb_new_page", help="Enter a title for the new page.")
if st.button("Add Page", help="Add a new page to your website.") and new_page_title:
    # Check for duplicate page names
    if new_page_title in page_titles:
        st.warning(f"A page named '{new_page_title}' already exists. Please use a different name.")
    else:
        pages.append({"title": new_page_title, "components": []})
        st.session_state.wb_current_page = len(pages) - 1
        st.rerun()
display_page_list(pages)
edited_title = edit_page_title(selected_page, selected_page_idx)
if edited_title != selected_page["title"]:
    # Check for duplicate page names when editing
    if edited_title in [p["title"] for p in pages if p != selected_page]:
        st.warning(f"A page named '{edited_title}' already exists. Please use a different name.")
    else:
        selected_page["title"] = edited_title

# 4. Components
st.subheader("4️⃣ Manage Components")
st.markdown("**Add Components:**")
avail_components = [c for c in COMPONENTS if c not in [comp["name"] if isinstance(comp, dict) else comp for comp in selected_page["components"]]]
comp_to_add = st.selectbox("Component:", ["-"] + avail_components, key="wb_add_comp", help="Select a component to add.")
if comp_to_add != "-" and st.button("Add Component", help="Add the selected component to this page."):
    selected_page["components"].append({
        "name": comp_to_add,
        "image": "",
        "icon": "",
        "props": {}
    })
    st.rerun()
# Normalize all components to dicts (migrate old string format)
for idx, comp in enumerate(selected_page["components"]):
    if isinstance(comp, str):
        selected_page["components"][idx] = {
            "name": comp,
            "image": "",
            "icon": "",
            "props": {}
        }
st.markdown("**Page Components:**")
display_component_list(selected_page["components"])

# 5. Advanced Code
st.subheader("5️⃣ Advanced: Edit HTML/CSS/JS")
st.markdown("Edit the global HTML, CSS, and JS for your website. This will be included in every page.")
st.session_state.wb_code["html"] = st.text_area("Custom HTML (body only):", value=st.session_state.wb_code["html"], height=120, help="Add custom HTML to the body of your website.")
st.session_state.wb_code["css"] = st.text_area("Custom CSS:", value=st.session_state.wb_code["css"], height=80, key="wb_css_editor", help="Add custom CSS for your website.")
st.session_state.wb_code["js"] = st.text_area("Custom JS:", value=st.session_state.wb_code["js"], height=80, key="wb_js_editor", help="Add custom JavaScript for your website.")

# 6. AI Assistant
st.subheader("6️⃣ 🤖 AI Website Assistant")
if "ai_code" not in st.session_state:
    st.session_state.ai_code = {"html": "", "css": "", "js": ""}
if "ai_prompt" not in st.session_state:
    st.session_state.ai_prompt = ""
st.session_state.ai_prompt = st.text_area("Describe what you want to add (e.g., 'Add a hero section with a blue background and a call-to-action button'):", value=st.session_state.ai_prompt, key="ai_prompt_input", help="Describe the section or feature you want the AI to generate.")
# Get API key from environment variables with fallback
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD6tgcbdvWEK8WhiLNku_K8ZEUtbJUAag8")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={GEMINI_API_KEY}"
def gemini_llm_generate_code(prompt):
    system_prompt = (
        "You are an expert web developer. Given a user prompt, generate a website section as three separate code blocks: HTML, CSS, and JS. "
        "Format your response as follows:\nHTML:\n<your html>\nCSS:\n<your css>\nJS:\n<your js>"
    )
    data = {
        "contents": [
            {"parts": [{"text": f"{system_prompt}\nUser prompt: {prompt}"}]}
        ]
    }
    try:
        st.info("Connecting to Gemini API...")
        response = requests.post(GEMINI_API_URL, json=data, timeout=60)
        
        if response.status_code != 200:
            st.error(f"API Error: Status code {response.status_code}")
            return {
                "html": f"<div>API Error: Status code {response.status_code}</div>", 
                "css": "/* API Error */", 
                "js": "// API Error"
            }
            
        response.raise_for_status()
        import re
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        html = re.search(r"HTML:\s*(.*?)\s*CSS:", text, re.DOTALL)
        css = re.search(r"CSS:\s*(.*?)\s*JS:", text, re.DOTALL)
        js = re.search(r"JS:\s*(.*)", text, re.DOTALL)
        return {
            "html": html.group(1).strip() if html else "<div>AI did not return HTML.</div>",
            "css": css.group(1).strip() if css else "/* AI did not return CSS */",
            "js": js.group(1).strip() if js else "// AI did not return JavaScript"
        }
    except requests.RequestException as e:
        st.error(f"Network error: {str(e)}. Please check your internet connection.")
        return {
            "html": f"<div class='error'>Network error: {str(e)}</div>", 
            "css": "/* Network error */", 
            "js": "// Network error"
        }
    except Exception as e:
        st.error(f"Error generating code: {str(e)}")
        error_details = traceback.format_exc()
        st.code(error_details, language="python")
        return {
            "html": f"<div class='error'>Error processing AI response: {str(e)}</div>", 
            "css": ".error{color:red;padding:10px;border:1px solid red;}", 
            "js": f"// Error: {str(e)}"
        }
if st.button("Generate with AI", key="ai_generate_btn", help="Generate code using Gemini AI."):
    with st.spinner("Generating code with Gemini..."):
        result = gemini_llm_generate_code(st.session_state.ai_prompt)
        st.session_state.ai_code = result
        add_ai_history(st.session_state.ai_prompt, result)
if st.session_state.ai_prompt_history:
    st.markdown("**Prompt History:**")
    for idx, prompt in enumerate(st.session_state.ai_prompt_history):
        st.markdown(f"{idx+1}. {prompt}")
        if st.button(f"Regenerate {idx+1}", key=f"ai_regen_{idx}", help="Regenerate code for this prompt"):
            with st.spinner("Regenerating with Gemini..."):
                result = gemini_llm_generate_code(prompt)
                st.session_state.ai_code = result
                add_ai_history(prompt, result)
st.text_area("AI Generated HTML", value=st.session_state.ai_code["html"], height=100, key="ai_html", help="HTML code generated by AI.")
st.text_area("AI Generated CSS", value=st.session_state.ai_code["css"], height=80, key="ai_css", help="CSS code generated by AI.")
st.text_area("AI Generated JS", value=st.session_state.ai_code["js"], height=80, key="ai_js", help="JS code generated by AI.")
st.markdown("**Preview:**")
ai_preview_html = f"""<style>{st.session_state.ai_code['css']}</style>{st.session_state.ai_code['html']}<script>{st.session_state.ai_code['js']}</script>"""
st.components.v1.html(ai_preview_html, height=300, scrolling=True)
def insert_ai_code_to_page():
    pages = st.session_state.wb_config["pages"]
    idx = st.session_state.wb_current_page
    ai_comp = {
        "name": "Custom Code (AI)",
        "image": "",
        "icon": "fa-solid fa-robot",
        "props": {
            "custom_code": {
                "html": st.session_state.ai_code["html"],
                "css": st.session_state.ai_code["css"],
                "js": st.session_state.ai_code["js"]
            }
        }
    }
    pages[idx]["components"].append(ai_comp)
    st.success("AI-generated code added as a new component!")
    st.rerun()
if st.button("Insert into Page", key="ai_insert_btn", help="Insert the generated code as a new component in the current page."):
    insert_ai_code_to_page()

# 7. Preview
st.subheader("7️⃣ Live Preview")
st.markdown("Preview your website in real time. Use the device toggles to see how it looks on different screens.")
preview_modes = {"Mobile": 375, "Tablet": 768, "Desktop": "100%"}
if "wb_preview_mode" not in st.session_state:
    st.session_state.wb_preview_mode = "Desktop"
selected_mode = st.radio("Preview Mode:", list(preview_modes.keys()), horizontal=True, index=list(preview_modes.keys()).index(st.session_state.wb_preview_mode), key="wb_preview_mode_radio", help="Select a device to preview your website.")
st.session_state.wb_preview_mode = selected_mode
preview_width = preview_modes[selected_mode]
render_html_preview(st.session_state.wb_config, st.session_state.wb_code, preview_width)

# 8. Export/Deploy
st.subheader("8️⃣ Export & Deploy")
st.markdown("Export your website as a ZIP file and follow the instructions to deploy it online.")
if st.button("Export as ZIP", help="Export your website as a ZIP file for deployment."):
    zip_file = generate_website_zip(st.session_state.wb_config, st.session_state.wb_code)
    st.download_button(
        label="Download Website ZIP",
        data=zip_file,
        file_name=f"{st.session_state.wb_config['title'].replace(' ', '_').lower()}_website.zip",
        mime="application/zip"
    )
st.markdown("---")
st.subheader("Deploy to GitHub Pages or Netlify")
st.markdown("""
**To deploy your exported website:**

**GitHub Pages:**
1. Unzip your exported website ZIP file.
2. Create a new repository on [GitHub](https://github.com/new).
3. Upload all files (including `index.html`, `styles.css`, etc.) to the repository.
4. Go to **Settings > Pages** in your repo, and set the source to the `main` branch (or `docs/` folder if you prefer).
5. Visit the provided GitHub Pages URL to see your site live!

[GitHub Pages Docs](https://docs.github.com/en/pages/getting-started-with-github-pages)

**Netlify:**
1. Go to [Netlify](https://app.netlify.com/) and sign up/log in.
2. Click **Add new site > Deploy manually**.
3. Drag and drop your unzipped website folder.
4. Netlify will give you a live URL instantly.

[Netlify Docs](https://docs.netlify.com/site-deploys/create-deploys/)
""")

# 9. Project Management
st.subheader("9️⃣ Project Management")
st.markdown("Save and load your website projects.")
project_dir = Path("storage/websites")
project_dir.mkdir(parents=True, exist_ok=True)
if st.button("Save Website Project", help="Save your current website project."):
    fname = f"{st.session_state.wb_config['title'].replace(' ', '_').lower()}.json"
    with open(project_dir / fname, "w", encoding="utf-8") as f:
        json.dump({"config": st.session_state.wb_config, "code": st.session_state.wb_code}, f, indent=2)
    st.success(f"Project saved as {fname}")
files = list(project_dir.glob("*.json"))
if files:
    fname = st.selectbox("Select project to load:", [f.name for f in files], key="wb_load_project", help="Select a project to load.")
    if st.button("Load Selected Project", help="Load the selected project."):
        try:
            with open(project_dir / fname, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Normalize page structure from saved data
                if "config" in data and "pages" in data["config"]:
                    data["config"] = normalize_page_structure(data["config"])
                st.session_state.wb_config = data["config"]
                st.session_state.wb_code = data["code"]
            st.success(f"Project '{fname}' loaded!")
            st.rerun()
        except Exception as e:
            st.error(f"Error loading project: {str(e)}")
else:
    st.info("No saved projects found.")

# 10. SEO & Metadata
st.subheader("🔟 SEO & Metadata")
st.markdown("Edit your website's SEO and metadata for better search engine visibility.")
if "metadata" not in st.session_state.wb_config:
    st.session_state.wb_config["metadata"] = {"site_title": st.session_state.wb_config["title"], "description": st.session_state.wb_config["description"], "favicon": ""}
meta = st.session_state.wb_config["metadata"]
meta["site_title"] = st.text_input("Site Title (for browser tab)", value=meta.get("site_title", ""), help="Set the title for your website (shown in browser tabs and search results).")
meta["description"] = st.text_area("Meta Description", value=meta.get("description", ""), help="Set a meta description for your website (improves SEO).")
favicon_url = st.text_input("Favicon URL (optional)", value=meta.get("favicon", ""), help="Set a favicon URL for your website.")

# Validate favicon URL
if favicon_url:
    if favicon_url.startswith(("http://", "https://")):
        try:
            response = requests.head(favicon_url, timeout=3)
            if response.status_code != 200:
                st.warning(f"Favicon URL returned status code {response.status_code}. The icon might not be available.")
            else:
                st.success("Favicon URL validated successfully.")
                meta["favicon"] = favicon_url
        except Exception:
            st.warning("Could not validate favicon URL. Please ensure it's a direct link to an image file.")
    else:
        st.warning("Please use a full URL starting with http:// or https://")
        favicon_url = ""
                
meta["favicon"] = favicon_url
st.session_state.wb_config["metadata"] = meta

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ using ALPHA Website Builder. Export and host your site anywhere!") 