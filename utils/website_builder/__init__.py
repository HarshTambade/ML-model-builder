"""
Website Builder module for ALPHA ML platform.
Provides tools to create and export websites through an intuitive UI.
"""

from .components import render_navbar, render_hero, render_text_block, render_button
from .components import render_card, render_image_gallery, render_form, render_divider
from .components import render_icon, render_columns, render_grid 
from .config import css_frameworks, color_schemes, website_templates, base_page_template
from .template_engine import render_html_preview, render_components, generate_website_zip

# Component renderers dictionary
COMPONENTS = {
    'navbar': render_navbar,
    'hero': render_hero,
    'text_block': render_text_block,
    'button': render_button,
    'card': render_card,
    'image_gallery': render_image_gallery,
    'form': render_form,
    'divider': render_divider,
    'icon': render_icon,
    'columns': render_columns,
    'grid': render_grid,
}

__all__ = [
    'COMPONENTS',
    'css_frameworks',
    'color_schemes',
    'website_templates',
    'base_page_template',
    'render_html_preview',
    'render_components',
    'generate_website_zip'
] 