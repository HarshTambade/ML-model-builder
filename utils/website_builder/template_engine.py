import streamlit as st
import os
import json
import shutil
import zipfile
import tempfile
import base64

def render_components(components, component_renderers):
    """
    Render HTML for components
    
    Args:
        components: List of component dictionaries
        component_renderers: Dictionary of component rendering functions
        
    Returns:
        string: HTML content for components
    """
    html = ""
    
    for component in components:
        component_type = component.get('type', '')
        content = component.get('content', {})
        
        if isinstance(component_type, str) and component_type in component_renderers:
            try:
                component_html = component_renderers[component_type](content)
                html += component_html
            except Exception as e:
                html += f'<div class="alert alert-danger">Error rendering {component_type}: {str(e)}</div>'
        else:
            # Handle unknown component types
            html += f'<div class="alert alert-warning">Unknown component type: {component_type}</div>'
    
    return html

def render_html_preview(website_config, css_frameworks, color_schemes, component_renderers):
    """
    Render HTML preview for the website builder
    
    Args:
        website_config: Dictionary containing website configuration
        css_frameworks: Dictionary mapping framework names to CDN URLs
        color_schemes: Dictionary mapping scheme names to color dictionaries
        component_renderers: Dictionary of component rendering functions
        
    Returns:
        string: HTML content for preview
    """
    # Extract configuration
    css_framework = website_config.get('css_framework', 'bootstrap')
    color_scheme = website_config.get('color_scheme', 'light')
    fonts = website_config.get('fonts', {
        'heading': 'Roboto, sans-serif',
        'body': 'Open Sans, sans-serif',
    })
    custom_css = website_config.get('custom_css', '')
    custom_js = website_config.get('custom_js', '')
    meta = website_config.get('meta', {
        'title': 'My Website',
        'description': 'A website built with ALPHA Website Builder',
        'keywords': 'website, builder, alpha',
        'author': '',
        'favicon': '',
    })
    
    # Get selected page or default to first page
    pages = website_config.get('pages', [{'name': 'Home', 'components': []}])
    selected_page_idx = st.session_state.get('selected_page_idx', 0)
    if selected_page_idx >= len(pages):
        selected_page_idx = 0
    selected_page = pages[selected_page_idx]
    
    # Generate navigation HTML
    nav_html = '<ul class="navbar-nav">\n'
    for i, page in enumerate(pages):
        active = 'active' if i == selected_page_idx else ''
        nav_html += f'<li class="nav-item"><a class="nav-link {active}" href="#">{page["name"]}</a></li>\n'
    nav_html += '</ul>\n'
    
    # Generate CSS variables for color scheme
    colors = color_schemes.get(color_scheme, color_schemes['light'])
    css_vars = f"""
    :root {{
        --primary-color: {colors['primary']};
        --secondary-color: {colors['secondary']};
        --background-color: {colors['background']};
        --text-color: {colors['text']};
        --heading-font: {fonts['heading']};
        --body-font: {fonts['body']};
    }}
    """
    
    # Get CSS framework URL
    css_framework_url = css_frameworks.get(css_framework, css_frameworks['bootstrap'])
    
    # Generate body content by rendering components
    body_content = render_components(selected_page.get('components', []), component_renderers)
    
    # Assemble complete HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{meta['title']}</title>
        <meta name="description" content="{meta['description']}">
        <meta name="keywords" content="{meta['keywords']}">
        <meta name="author" content="{meta['author']}">
        {f'<link rel="icon" href="{meta["favicon"]}">' if meta.get('favicon') else ''}
        
        <!-- CSS Framework -->
        <link rel="stylesheet" href="{css_framework_url}">
        
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Open+Sans:wght@300;400;700&family=Lato:wght@300;400;700&family=Montserrat:wght@300;400;700&family=Raleway:wght@300;400;700&family=Playfair+Display:wght@400;700&family=Merriweather:wght@300;400;700&display=swap" rel="stylesheet">
        
        <!-- Font Awesome for icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        
        <style>
            /* Custom CSS Variables */
            {css_vars}
            
            /* Base Styles */
            body {{
                font-family: var(--body-font);
                color: var(--text-color);
                background-color: var(--background-color);
                line-height: 1.6;
            }}
            
            h1, h2, h3, h4, h5, h6 {{
                font-family: var(--heading-font);
                color: var(--primary-color);
            }}
            
            .btn-primary {{
                background-color: var(--primary-color);
                border-color: var(--primary-color);
            }}
            
            .btn-secondary {{
                background-color: var(--secondary-color);
                border-color: var(--secondary-color);
            }}
            
            /* Custom CSS */
            {custom_css}
        </style>
    </head>
    <body>
        {body_content}
        
        <!-- JS Framework -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.min.js"></script>
        
        <!-- Custom JavaScript -->
        <script>
            {custom_js}
        </script>
    </body>
    </html>
    """
    
    return html

def generate_website_zip(website_config, css_frameworks, color_schemes, component_renderers):
    """
    Generate a ZIP file with the website files
    
    Args:
        website_config: Dictionary containing website configuration
        css_frameworks: Dictionary mapping framework names to CDN URLs
        color_schemes: Dictionary mapping scheme names to color dictionaries
        component_renderers: Dictionary of component rendering functions
        
    Returns:
        bytes: ZIP file content
    """
    # Create a temporary directory to store website files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create index.html and page HTML files
        for i, page in enumerate(website_config['pages']):
            # Save current selected page index
            original_selected_idx = st.session_state.get('selected_page_idx', 0)
            
            # Temporarily set selected page for rendering
            st.session_state.selected_page_idx = i
            
            # Generate HTML for this page
            html_content = render_html_preview(
                website_config, 
                css_frameworks, 
                color_schemes, 
                component_renderers
            )
            
            # Determine filename (index.html for first page, page_name.html for others)
            filename = 'index.html' if i == 0 else f"{page['name'].lower().replace(' ', '_')}.html"
            
            # Write HTML file
            with open(os.path.join(temp_dir, filename), 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Restore original selected page index
            st.session_state.selected_page_idx = original_selected_idx
        
        # Create CSS file
        if website_config.get('custom_css'):
            with open(os.path.join(temp_dir, 'styles.css'), 'w', encoding='utf-8') as f:
                f.write(website_config['custom_css'])
        
        # Create JS file
        if website_config.get('custom_js'):
            with open(os.path.join(temp_dir, 'scripts.js'), 'w', encoding='utf-8') as f:
                f.write(website_config['custom_js'])
        
        # Create a ZIP file
        zip_path = os.path.join(temp_dir, 'website.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # Add all files from the temp directory to the ZIP
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file != 'website.zip':  # Skip the ZIP file itself
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, temp_dir)
                        zipf.write(file_path, arcname)
        
        # Read the ZIP file
        with open(zip_path, 'rb') as f:
            return f.read() 