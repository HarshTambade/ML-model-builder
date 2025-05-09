"""
Configuration settings for the ALPHA Website Builder.
Contains CSS frameworks, color schemes, and base page templates.
"""

# Available CSS frameworks with their CDN URLs
css_frameworks = {
    'bootstrap': 'https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css',
    'bulma': 'https://cdn.jsdelivr.net/npm/bulma@0.9.4/css/bulma.min.css',
    'tailwind': 'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
    'material': 'https://unpkg.com/materialize-css@1.0.0/dist/css/materialize.min.css',
    'foundation': 'https://cdn.jsdelivr.net/npm/foundation-sites@6.7.5/dist/css/foundation.min.css'
}

# Available color schemes with their color values
color_schemes = {
    'light': {
        'primary': '#007bff',
        'secondary': '#6c757d',
        'background': '#ffffff',
        'text': '#212529'
    },
    'dark': {
        'primary': '#0d6efd',
        'secondary': '#6c757d',
        'background': '#212529',
        'text': '#f8f9fa'
    },
    'earth': {
        'primary': '#2e7d32',
        'secondary': '#795548',
        'background': '#f5f5f5',
        'text': '#424242'
    },
    'ocean': {
        'primary': '#0277bd',
        'secondary': '#00838f',
        'background': '#e1f5fe',
        'text': '#01579b'
    },
    'sunset': {
        'primary': '#ef6c00',
        'secondary': '#d32f2f',
        'background': '#fff8e1',
        'text': '#3e2723'
    },
    'monochrome': {
        'primary': '#424242',
        'secondary': '#757575',
        'background': '#fafafa',
        'text': '#212121'
    },
    'pastel': {
        'primary': '#9575cd',
        'secondary': '#4fc3f7',
        'background': '#f3e5f5',
        'text': '#37474f'
    },
    'neon': {
        'primary': '#00e676',
        'secondary': '#00b0ff',
        'background': '#212121',
        'text': '#ffffff'
    }
}

# Base template for empty pages
base_page_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <meta name="description" content="{{description}}">
    <meta name="keywords" content="{{keywords}}">
    <meta name="author" content="{{author}}">
    {{favicon}}
    
    <!-- CSS Framework -->
    <link rel="stylesheet" href="{{css_framework_url}}">
    
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family={{heading_font}}&family={{body_font}}&display=swap" rel="stylesheet">
    
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        /* Custom CSS Variables */
        :root {
            --primary-color: {{primary_color}};
            --secondary-color: {{secondary_color}};
            --background-color: {{background_color}};
            --text-color: {{text_color}};
            --heading-font: {{heading_font}};
            --body-font: {{body_font}};
        }
        
        /* Base Styles */
        body {
            font-family: var(--body-font);
            color: var(--text-color);
            background-color: var(--background-color);
            line-height: 1.6;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--heading-font);
            color: var(--primary-color);
        }
        
        .btn-primary {
            background-color: var(--primary-color);
            border-color: var(--primary-color);
        }
        
        .btn-secondary {
            background-color: var(--secondary-color);
            border-color: var(--secondary-color);
        }
        
        /* Custom CSS */
        {{custom_css}}
    </style>
</head>
<body>
    {{body_content}}
    
    <!-- Framework JS -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.min.js"></script>
    
    <!-- Custom JavaScript -->
    <script>
        {{custom_js}}
    </script>
</body>
</html>
"""

# Templates for different website types
website_templates = {
    'blank': {
        'name': 'Blank Canvas',
        'description': 'Start from scratch',
        'preview_image': 'https://via.placeholder.com/300x200?text=Blank+Canvas',
        'pages': [
            {'name': 'Home', 'components': []}
        ]
    },
    'business': {
        'name': 'Business Site',
        'description': 'Professional layout for businesses',
        'preview_image': 'https://via.placeholder.com/300x200?text=Business+Template',
        'pages': [
            {'name': 'Home', 'components': [
                {'type': 'navbar', 'content': {'brand': 'Business Name', 'links': ['Home', 'About', 'Services', 'Contact']}},
                {'type': 'hero', 'content': {'title': 'Welcome to Our Business', 'subtitle': 'We provide the best services', 'cta': 'Learn More'}},
                {'type': 'text_block', 'content': {'text': 'We are a professional business providing top-notch services to our clients.'}}
            ]},
            {'name': 'About', 'components': [
                {'type': 'navbar', 'content': {'brand': 'Business Name', 'links': ['Home', 'About', 'Services', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'About our company and our mission.'}}
            ]},
            {'name': 'Services', 'components': [
                {'type': 'navbar', 'content': {'brand': 'Business Name', 'links': ['Home', 'About', 'Services', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'Services we offer.'}}
            ]},
            {'name': 'Contact', 'components': [
                {'type': 'navbar', 'content': {'brand': 'Business Name', 'links': ['Home', 'About', 'Services', 'Contact']}},
                {'type': 'form', 'content': {'fields': [
                    {'type': 'text', 'label': 'Name'},
                    {'type': 'email', 'label': 'Email'},
                    {'type': 'textarea', 'label': 'Message'}
                ], 'submit_text': 'Send'}}
            ]}
        ]
    },
    'portfolio': {
        'name': 'Portfolio',
        'description': 'Showcase your work',
        'preview_image': 'https://via.placeholder.com/300x200?text=Portfolio+Template',
        'pages': [
            {'name': 'Home', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Portfolio', 'links': ['Home', 'Projects', 'About Me', 'Contact']}},
                {'type': 'hero', 'content': {'title': 'John Doe', 'subtitle': 'Designer & Developer', 'cta': 'View My Work'}},
                {'type': 'image_gallery', 'content': {'images': [
                    'https://via.placeholder.com/300x300?text=Project+1', 
                    'https://via.placeholder.com/300x300?text=Project+2', 
                    'https://via.placeholder.com/300x300?text=Project+3'
                ]}}
            ]},
            {'name': 'Projects', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Portfolio', 'links': ['Home', 'Projects', 'About Me', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'My projects showcase.'}}
            ]},
            {'name': 'About Me', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Portfolio', 'links': ['Home', 'Projects', 'About Me', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'About me and my skills.'}}
            ]},
            {'name': 'Contact', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Portfolio', 'links': ['Home', 'Projects', 'About Me', 'Contact']}},
                {'type': 'form', 'content': {'fields': [
                    {'type': 'text', 'label': 'Name'},
                    {'type': 'email', 'label': 'Email'},
                    {'type': 'textarea', 'label': 'Message'}
                ], 'submit_text': 'Send'}}
            ]}
        ]
    },
    'blog': {
        'name': 'Blog',
        'description': 'Share your stories',
        'preview_image': 'https://via.placeholder.com/300x200?text=Blog+Template',
        'pages': [
            {'name': 'Home', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Blog', 'links': ['Home', 'Articles', 'About', 'Contact']}},
                {'type': 'hero', 'content': {'title': 'My Blog', 'subtitle': 'Thoughts, stories and ideas', 'cta': 'Read More'}},
                {'type': 'text_block', 'content': {'text': 'Welcome to my blog where I share my thoughts and ideas.'}}
            ]},
            {'name': 'Articles', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Blog', 'links': ['Home', 'Articles', 'About', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'List of articles.'}}
            ]},
            {'name': 'About', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Blog', 'links': ['Home', 'Articles', 'About', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'About me and this blog.'}}
            ]},
            {'name': 'Contact', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Blog', 'links': ['Home', 'Articles', 'About', 'Contact']}},
                {'type': 'form', 'content': {'fields': [
                    {'type': 'text', 'label': 'Name'},
                    {'type': 'email', 'label': 'Email'},
                    {'type': 'textarea', 'label': 'Message'}
                ], 'submit_text': 'Send'}}
            ]}
        ]
    },
    'ecommerce': {
        'name': 'E-Commerce',
        'description': 'Sell products online',
        'preview_image': 'https://via.placeholder.com/300x200?text=E-Commerce+Template',
        'pages': [
            {'name': 'Home', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Store', 'links': ['Home', 'Products', 'About', 'Contact']}},
                {'type': 'hero', 'content': {'title': 'Welcome to My Store', 'subtitle': 'Quality products at great prices', 'cta': 'Shop Now'}},
                {'type': 'grid', 'content': {'items': [
                    {'title': 'Product 1', 'content': 'Description and price'},
                    {'title': 'Product 2', 'content': 'Description and price'},
                    {'title': 'Product 3', 'content': 'Description and price'},
                    {'title': 'Product 4', 'content': 'Description and price'}
                ]}}
            ]},
            {'name': 'Products', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Store', 'links': ['Home', 'Products', 'About', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'Our products catalog.'}}
            ]},
            {'name': 'About', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Store', 'links': ['Home', 'Products', 'About', 'Contact']}},
                {'type': 'text_block', 'content': {'text': 'About our store and our mission.'}}
            ]},
            {'name': 'Contact', 'components': [
                {'type': 'navbar', 'content': {'brand': 'My Store', 'links': ['Home', 'Products', 'About', 'Contact']}},
                {'type': 'form', 'content': {'fields': [
                    {'type': 'text', 'label': 'Name'},
                    {'type': 'email', 'label': 'Email'},
                    {'type': 'textarea', 'label': 'Message'}
                ], 'submit_text': 'Send'}}
            ]}
        ]
    },
    'landing': {
        'name': 'Landing Page',
        'description': 'Conversion-focused single page',
        'preview_image': 'https://via.placeholder.com/300x200?text=Landing+Page+Template',
        'pages': [
            {'name': 'Home', 'components': [
                {'type': 'navbar', 'content': {'brand': 'Product Name', 'links': ['Features', 'Pricing', 'Contact']}},
                {'type': 'hero', 'content': {'title': 'Your Amazing Product', 'subtitle': 'Solve your problems with our solution', 'cta': 'Get Started'}},
                {'type': 'text_block', 'content': {'text': 'Our product helps you achieve your goals faster and more efficiently.'}},
                {'type': 'grid', 'content': {'items': [
                    {'title': 'Feature 1', 'content': 'Description of feature 1'},
                    {'title': 'Feature 2', 'content': 'Description of feature 2'},
                    {'title': 'Feature 3', 'content': 'Description of feature 3'},
                    {'title': 'Feature 4', 'content': 'Description of feature 4'}
                ]}},
                {'type': 'form', 'content': {'fields': [
                    {'type': 'text', 'label': 'Name'},
                    {'type': 'email', 'label': 'Email'},
                    {'type': 'textarea', 'label': 'Message'}
                ], 'submit_text': 'Sign Up'}}
            ]}
        ]
    }
} 