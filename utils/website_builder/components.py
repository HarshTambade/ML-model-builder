"""
Component rendering functions for the ALPHA Website Builder.
Each function takes a content dictionary and returns HTML markup.
"""

def render_navbar(content):
    """Render a navigation bar component"""
    brand = content.get('brand', 'Brand')
    links = content.get('links', ['Home', 'About', 'Contact'])
    
    links_html = ""
    for link in links:
        links_html += f'<li class="nav-item"><a class="nav-link" href="#">{link}</a></li>'
    
    html = f"""
    <nav class="navbar navbar-expand-lg navbar-light bg-light mb-4">
        <div class="container">
            <a class="navbar-brand" href="#">{brand}</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav ms-auto">
                    {links_html}
                </ul>
            </div>
        </div>
    </nav>
    """
    return html

def render_hero(content):
    """Render a hero section component"""
    title = content.get('title', 'Welcome')
    subtitle = content.get('subtitle', 'This is a hero section')
    cta = content.get('cta', 'Learn More')
    
    html = f"""
    <section class="py-5 text-center container">
        <div class="row py-lg-5">
            <div class="col-lg-8 col-md-10 mx-auto">
                <h1 class="fw-bold">{title}</h1>
                <p class="lead text-muted">{subtitle}</p>
                <p>
                    <a href="#" class="btn btn-primary my-2">{cta}</a>
                </p>
            </div>
        </div>
    </section>
    """
    return html

def render_text_block(content):
    """Render a text block component"""
    text = content.get('text', 'Enter your text here.')
    
    # Convert newlines to <br> tags
    text = text.replace('\n', '<br>')
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            <div class="col">
                <p>{text}</p>
            </div>
        </div>
    </div>
    """
    return html

def render_button(content):
    """Render a button component"""
    text = content.get('text', 'Click Me')
    url = content.get('url', '#')
    style = content.get('style', 'primary')
    
    html = f"""
    <div class="container my-3">
        <div class="row">
            <div class="col">
                <a href="{url}" class="btn btn-{style}">{text}</a>
            </div>
        </div>
    </div>
    """
    return html

def render_card(content):
    """Render a card component"""
    title = content.get('title', 'Card Title')
    text = content.get('text', 'Card content goes here')
    image = content.get('image', '')
    
    image_html = f'<img src="{image}" class="card-img-top" alt="{title}">' if image else ''
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            <div class="col">
                <div class="card" style="width: 18rem;">
                    {image_html}
                    <div class="card-body">
                        <h5 class="card-title">{title}</h5>
                        <p class="card-text">{text}</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return html

def render_image_gallery(content):
    """Render an image gallery component"""
    images = content.get('images', [])
    
    # Create image cards
    image_html = ""
    for img in images:
        image_html += f"""
        <div class="col-md-4 mb-4">
            <div class="card">
                <img src="{img}" class="img-fluid rounded" alt="Gallery Image">
            </div>
        </div>
        """
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            {image_html}
        </div>
    </div>
    """
    return html

def render_form(content):
    """Render a form component"""
    fields = content.get('fields', [])
    submit_text = content.get('submit_text', 'Submit')
    
    fields_html = ""
    for field in fields:
        field_type = field.get('type', 'text')
        label = field.get('label', 'Field')
        field_id = label.lower().replace(' ', '_')
        
        if field_type == 'textarea':
            fields_html += f"""
            <div class="mb-3">
                <label for="{field_id}" class="form-label">{label}</label>
                <textarea class="form-control" id="{field_id}" rows="3"></textarea>
            </div>
            """
        else:
            fields_html += f"""
            <div class="mb-3">
                <label for="{field_id}" class="form-label">{label}</label>
                <input type="{field_type}" class="form-control" id="{field_id}">
            </div>
            """
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            <div class="col-md-8">
                <form>
                    {fields_html}
                    <button type="submit" class="btn btn-primary">{submit_text}</button>
                </form>
            </div>
        </div>
    </div>
    """
    return html

def render_divider(content):
    """Render a divider component"""
    html = """
    <div class="container my-4">
        <hr>
    </div>
    """
    return html

def render_icon(content):
    """Render an icon component"""
    name = content.get('name', 'star')
    size = content.get('size', '2x')
    color = content.get('color', '#007bff')
    
    html = f"""
    <div class="container my-3 text-center">
        <i class="fas fa-{name} fa-{size}" style="color: {color};"></i>
    </div>
    """
    return html

def render_columns(content):
    """Render a columns layout component"""
    columns_data = content.get('columns', [])
    
    columns_html = ""
    col_count = len(columns_data)
    col_width = 12 // col_count if col_count > 0 else 12
    
    for column in columns_data:
        # Render components inside this column
        column_components = column.get('components', [])
        column_content = ""
        
        for component in column_components:
            component_type = component.get('type', '')
            component_content = component.get('content', {})
            
            # This will be handled by the render_components function from template_engine
            # Just add a placeholder for now
            column_content += f'<div class="component-placeholder" data-type="{component_type}"></div>'
        
        # If no components, add placeholder
        if not column_content:
            column_content = '<div class="p-3 border bg-light">Column content</div>'
        
        columns_html += f"""
        <div class="col-md-{col_width}">
            {column_content}
        </div>
        """
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            {columns_html}
        </div>
    </div>
    """
    return html

def render_grid(content):
    """Render a grid layout component"""
    items = content.get('items', [])
    
    items_html = ""
    for item in items:
        title = item.get('title', 'Item')
        item_content = item.get('content', '')
        
        items_html += f"""
        <div class="col-md-6 col-lg-3 mb-4">
            <div class="card h-100">
                <div class="card-body">
                    <h5 class="card-title">{title}</h5>
                    <p class="card-text">{item_content}</p>
                </div>
            </div>
        </div>
        """
    
    html = f"""
    <div class="container my-4">
        <div class="row">
            {items_html}
        </div>
    </div>
    """
    return html 