"""
Helper utility functions for the ALPHA platform.
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
import base64
from typing import Optional, Union, List, Dict, Any

@st.cache_data
def load_css(css_file: str) -> None:
    """
    Load and inject CSS from a file into the Streamlit app.
    
    Args:
        css_file: Path to the CSS file relative to the application root
    """
    try:
        if os.path.exists(css_file):
            with open(css_file, "r") as f:
                css = f.read()
                st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        else:
            st.warning(f"CSS file not found: {css_file}")
    except Exception as e:
        st.error(f"Error loading CSS file: {str(e)}")

@st.cache_data
def get_image_as_base64(file_path: str) -> Optional[str]:
    """
    Convert an image file to a base64 encoded string.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Base64 encoded string or None if file not found
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        else:
            st.warning(f"Image file not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def create_download_link(object_to_download: Union[str, bytes], 
                         download_filename: str, 
                         button_text: str = 'Download') -> str:
    """
    Create a download link for any object that can be converted to bytes.
    
    Args:
        object_to_download: The object to be downloaded
        download_filename: Filename for the download
        button_text: Text to display on the download button
        
    Returns:
        HTML string containing the download link
    """
    try:
        # Convert object to bytes if it's not already
        if isinstance(object_to_download, str):
            object_to_download = object_to_download.encode()
            
        b64 = base64.b64encode(object_to_download).decode()
        
        # Determine MIME type based on file extension
        mime_type = "application/octet-stream"  # Default
        extension = download_filename.split('.')[-1].lower()
        
        if extension in ['csv']:
            mime_type = "text/csv"
        elif extension in ['json']:
            mime_type = "application/json"
        elif extension in ['txt']:
            mime_type = "text/plain"
        elif extension in ['pdf']:
            mime_type = "application/pdf"
        elif extension in ['png']:
            mime_type = "image/png"
        elif extension in ['jpg', 'jpeg']:
            mime_type = "image/jpeg"
        elif extension in ['zip']:
            mime_type = "application/zip"
            
        download_link = f'<a href="data:{mime_type};base64,{b64}" download="{download_filename}">{button_text}</a>'
        return download_link
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return ""

def format_large_number(num: Union[int, float]) -> str:
    """
    Format large numbers with K, M, B suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string representation of the number
    """
    if num is None:
        return "N/A"
        
    try:
        num = float(num)
        magnitude = 0
        
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
            
        suffix = ['', 'K', 'M', 'B', 'T'][min(magnitude, 4)]
        
        if magnitude == 0:
            return f"{num:.0f}{suffix}"
        else:
            return f"{num:.1f}{suffix}"
    except:
        return str(num)

@st.cache_data
def truncate_text(text: str, max_length: int = 100, add_ellipsis: bool = True) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of the truncated text
        add_ellipsis: Whether to add "..." at the end of truncated text
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
        
    truncated = text[:max_length].rstrip()
    if add_ellipsis:
        truncated += "..."
        
    return truncated

def display_info_card(title: str, content: str, icon: str = "info-circle", 
                     color: str = "#0066cc", bg_color: str = "#f8f9fa") -> None:
    """
    Display an information card with a title, content, and icon.
    
    Args:
        title: Card title
        content: Card content text
        icon: Font Awesome icon name (without the 'fa-' prefix)
        color: Text color (hex or CSS color name)
        bg_color: Background color (hex or CSS color name)
    """
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 5px; background-color: {bg_color}; margin-bottom: 15px;">
        <h4 style="color: {color}; margin-top: 0;">
            <i class="fas fa-{icon}" style="margin-right: 10px;"></i>{title}
        </h4>
        <p style="margin-bottom: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_responsive_columns(ratios: List[int]) -> List[Any]:
    """
    Create responsive columns with specified width ratios.
    Ensures the columns adjust well on different screen sizes.
    
    Args:
        ratios: List of relative width ratios
        
    Returns:
        List of column objects
    """
    # Calculate the total ratio sum
    total = sum(ratios)
    
    # Convert ratios to decimal percentages
    widths = [r/total for r in ratios]
    
    # Create columns with specified widths
    return st.columns(ratios)

def create_custom_table(data: List[Dict[str, Any]], 
                      headers: Optional[List[str]] = None,
                      col_widths: Optional[List[str]] = None) -> None:
    """
    Create a custom HTML table with more styling options than st.table().
    
    Args:
        data: List of dictionaries containing table data
        headers: Column headers (uses dict keys if None)
        col_widths: List of CSS width values for columns
    """
    if not data:
        st.warning("No data to display in table")
        return
        
    # Use dictionary keys as headers if not provided
    if headers is None:
        headers = list(data[0].keys())
        
    # Set default column widths if not provided
    if col_widths is None:
        col_widths = [f"{100/len(headers)}%" for _ in headers]
        
    # Create the table header
    header_html = "".join([f"<th style='width: {width};'>{header}</th>" 
                          for header, width in zip(headers, col_widths)])
    
    # Create table rows
    rows_html = ""
    for row in data:
        row_html = "".join([f"<td>{row.get(header, '')}</td>" for header in headers])
        rows_html += f"<tr>{row_html}</tr>"
    
    # Assemble the complete table
    table_html = f"""
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr>{header_html}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """
    
    # Add custom CSS
    table_css = """
    <style>
        table {
            border: 1px solid #ddd;
            border-collapse: collapse;
            width: 100%;
        }
        th {
            background-color: #f2f2f2;
            padding: 8px;
            text-align: left;
            border: 1px solid #ddd;
            font-weight: bold;
        }
        td {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
    </style>
    """
    
    # Display the table
    st.markdown(table_css + table_html, unsafe_allow_html=True)

@st.cache_data
def generate_text_summary(text: str, max_length: int = 200) -> str:
    """
    Generate a summary of text content.
    Implements a simple extractive summarization approach.
    
    Args:
        text: Text to summarize
        max_length: Maximum length of summary in characters
        
    Returns:
        Summarized text
    """
    if not text or len(text) <= max_length:
        return text
        
    # Split text into sentences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= 3:
        return truncate_text(text, max_length)
        
    # Simple scoring: prioritize first and last sentences
    scores = []
    for i, sentence in enumerate(sentences):
        score = 0
        # First sentences likely contain important info
        if i == 0:
            score += 5
        elif i == 1:
            score += 3
        # Last sentences often contain conclusions
        elif i == len(sentences) - 1:
            score += 4
        elif i == len(sentences) - 2:
            score += 2
        
        # Longer sentences might contain more information (but cap the bonus)
        length_score = min(len(sentence) / 20, 3)
        score += length_score
        
        # Sentences with numbers often contain facts
        if re.search(r'\d+', sentence):
            score += 2
            
        scores.append((score, sentence))
    
    # Sort by score and take top sentences
    scores.sort(reverse=True)
    top_sentences = [s[1] for s in scores[:3]]
    
    # Re-sort sentences to maintain original order
    original_order = []
    for sentence in top_sentences:
        original_order.append((sentences.index(sentence), sentence))
    
    original_order.sort()
    summary = ' '.join([s[1] for s in original_order])
    
    # Truncate if still too long
    if len(summary) > max_length:
        summary = truncate_text(summary, max_length)
        
    return summary 