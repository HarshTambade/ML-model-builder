"""
ALPHA - End-to-End Machine Learning Platform
Natural Language Analysis Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Import utility modules
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)
from utils.data import get_dataset_list, get_dataset_info, load_dataset, get_dataset_by_id
from utils.imports import is_package_available, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Natural Language Analysis")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Natural Language Analysis",
    description="Analyze text data with natural language processing",
    icon="💬"
)

# Check for NLP libraries
nltk_available = is_package_available("nltk")
spacy_available = is_package_available("spacy")
transformers_available = is_package_available("transformers")

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for NLP analysis. Please install pandas.')
    st.stop()
if not is_package_available('numpy'):
    st.warning('NumPy is not available. Some features may not work.')

# Simple NLP functions - Basic analysis that doesn't require external libraries
def count_words(text):
    """Count the number of words in text."""
    if not text:
        return 0
    words = re.findall(r'\w+', text.lower())
    return len(words)

def get_word_frequencies(text, top_n=20):
    """Get the most frequent words in text."""
    if not text:
        return {}
    words = re.findall(r'\w+', text.lower())
    # Filter out short words (typically less meaningful)
    words = [word for word in words if len(word) > 2]
    return Counter(words).most_common(top_n)

def get_sentence_count(text):
    """Count the number of sentences in text."""
    if not text:
        return 0
    # Simple sentence counting based on punctuation
    sentences = re.split(r'[.!?]+', text)
    # Remove empty sentences
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)

def calculate_readability(text):
    """Calculate simple readability metrics."""
    if not text:
        return {"avg_word_length": 0, "avg_sentence_length": 0}
    
    words = re.findall(r'\w+', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    
    if not words or not sentences:
        return {"avg_word_length": 0, "avg_sentence_length": 0}
    
    avg_word_length = sum(len(word) for word in words) / len(words)
    avg_sentence_length = len(words) / len(sentences)
    
    return {
        "avg_word_length": round(avg_word_length, 2),
        "avg_sentence_length": round(avg_sentence_length, 2)
    }

# Create tabs
nlp_tabs = create_tab_panels("Text Analysis", "Text Statistics", "Word Cloud", "Sentiment Analysis")

# Tab 1: Text Analysis
with nlp_tabs[0]:
    st.markdown("### Text Analysis")
    
    # Text input methods
    input_method = st.radio(
        "Choose input method:",
        ["Enter text", "Upload text file", "Select dataset column"]
    )
    
    text_input = ""
    
    if input_method == "Enter text":
        text_input = st.text_area(
            "Enter text to analyze:",
            height=200,
            placeholder="Enter or paste your text here..."
        )
    
    elif input_method == "Upload text file":
        uploaded_file = st.file_uploader("Upload a text file", type=["txt", "csv", "md"])
        if uploaded_file:
            try:
                text_input = uploaded_file.getvalue().decode("utf-8")
                st.success(f"File uploaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    elif input_method == "Select dataset column":
        # Get available datasets
        datasets = get_dataset_list()
        if not datasets:
            st.info("No datasets available. Please import or create a dataset first.")
        else:
            selected_dataset = st.selectbox("Select a dataset", datasets)
            
            if selected_dataset:
                # Load the dataset
                dataset_result = load_dataset(selected_dataset)
                
                if isinstance(dataset_result, dict) and "error" in dataset_result:
                    st.error(f"Error loading dataset: {dataset_result['error']}")
                else:
                    df = dataset_result
                    # Get text columns (object dtypes which might contain text)
                    text_columns = df.select_dtypes(include=['object']).columns.tolist()
                    
                    if not text_columns:
                        st.warning("No text columns found in the selected dataset.")
                    else:
                        selected_column = st.selectbox("Select text column", text_columns)
                        
                        if selected_column:
                            # Show sample
                            st.markdown("#### Sample data:")
                            st.dataframe(df[selected_column].head())
                            
                            # Concatenate all text for analysis
                            text_input = " ".join(df[selected_column].astype(str).tolist())
                            
                            # Option to analyze individual rows
                            analyze_individual = st.checkbox("Analyze individual rows instead of combined text")
                            
                            if analyze_individual:
                                row_index = st.slider("Select row to analyze", 0, len(df)-1, 0)
                                text_input = df[selected_column].iloc[row_index]
                                st.markdown(f"**Analyzing row {row_index}:**")
                                st.markdown(f"> {text_input}")
    
    # Analyze button
    if text_input:
        st.success("Text loaded successfully!")
        
        if st.button("Analyze Text"):
            with st.spinner("Analyzing..."):
                # Perform text analysis
                word_count = count_words(text_input)
                sentence_count = get_sentence_count(text_input)
                readability = calculate_readability(text_input)
                word_freq = get_word_frequencies(text_input)
                
                # Display results
                st.markdown("### Analysis Results")
                
                # Basic stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", word_count)
                with col2:
                    st.metric("Sentence Count", sentence_count)
                with col3:
                    st.metric("Avg. Sentence Length", f"{readability['avg_sentence_length']} words")
                
                # Word frequency
                st.markdown("#### Most Common Words")
                
                # Create a clean dataframe of word frequencies
                word_freq_df = pd.DataFrame(word_freq, columns=["Word", "Frequency"])
                
                # Display as table and chart
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.dataframe(word_freq_df)
                
                with col2:
                    # Create and display bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Limit to top 10 for better visualization
                    plot_data = word_freq_df.head(10)
                    
                    # Plot horizontally for better readability of words
                    ax.barh(plot_data["Word"], plot_data["Frequency"])
                    ax.set_title("Top 10 Most Frequent Words")
                    ax.set_xlabel("Frequency")
                    
                    # Display the plot
                    st.pyplot(fig)

# Tab 2: Text Statistics
with nlp_tabs[1]:
    st.markdown("### Text Statistics")
    
    st.markdown("""
    This tab provides statistical analysis of text data. 
    Please use the Text Analysis tab to input your text and run the analysis.
    """)
    
    if "word_freq_df" in locals():
        st.success("Text analysis results are available!")
        
        # Calculate additional statistics
        unique_words = len(set(re.findall(r'\w+', text_input.lower())))
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Words", word_count)
            st.metric("Unique Words", unique_words)
        
        with col2:
            st.metric("Sentences", sentence_count)
            st.metric("Avg. Word Length", f"{readability['avg_word_length']} chars")
        
        with col3:
            lexical_diversity = round(unique_words / word_count, 3) if word_count > 0 else 0
            st.metric("Lexical Diversity", lexical_diversity)
            st.metric("Avg. Sentence Length", f"{readability['avg_sentence_length']} words")
        
        # Word length distribution
        st.markdown("#### Word Length Distribution")
        words = re.findall(r'\w+', text_input.lower())
        word_lengths = [len(word) for word in words]
        
        # Create distribution
        length_count = Counter(word_lengths)
        
        # Convert to dataframe
        length_df = pd.DataFrame(sorted(length_count.items()), columns=["Word Length", "Count"])
        
        # Display as chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(length_df["Word Length"], length_df["Count"])
        ax.set_title("Word Length Distribution")
        ax.set_xlabel("Word Length (characters)")
        ax.set_ylabel("Count")
        
        st.pyplot(fig)
    else:
        st.info("No analysis results available. Please run text analysis first.")

# Tab 3: Word Cloud
with nlp_tabs[2]:
    st.markdown("### Word Cloud Visualization")
    
    if not is_package_available("wordcloud"):
        st.warning("The WordCloud package is not available. Install with: pip install wordcloud")
        
        # Show a placeholder word cloud image
        st.markdown("""
        A word cloud would visually represent the most frequent words in your text.
        More frequent words appear larger in the visualization.
        
        Please install the wordcloud package to enable this feature.
        """)
    else:
        # If we have the wordcloud package and text has been analyzed
        if "text_input" in locals() and text_input:
            from wordcloud import WordCloud
            
            st.markdown("#### Generate Word Cloud")
            
            # Options for word cloud
            min_word_length = st.slider("Minimum word length", 1, 10, 3)
            max_words = st.slider("Maximum number of words", 50, 500, 200)
            
            if st.button("Generate Word Cloud"):
                with st.spinner("Generating word cloud..."):
                    # Remove short words
                    cleaned_text = ' '.join([word for word in text_input.lower().split() if len(word) >= min_word_length])
                    
                    # Generate word cloud
                    wordcloud = WordCloud(
                        width=800, 
                        height=400,
                        background_color='white',
                        max_words=max_words,
                        contour_width=3
                    ).generate(cleaned_text)
                    
                    # Display the word cloud
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    
                    st.pyplot(fig)
                    
                    # Option to download
                    st.markdown("Right-click on the image to save it.")
        else:
            st.info("Please enter or upload text in the Text Analysis tab first.")

# Tab 4: Sentiment Analysis
with nlp_tabs[3]:
    st.markdown("### Sentiment Analysis")
    
    if not transformers_available:
        st.warning("Advanced sentiment analysis requires the transformers library. Basic sentiment analysis will be used instead.")
        
        # Simple lexicon-based sentiment
        if "text_input" in locals() and text_input:
            st.markdown("#### Basic Sentiment Analysis")
            
            st.info("This is a basic sentiment analysis using a simple lexicon-based approach. For more accurate results, install the transformers library.")
            
            # Very simple sentiment analysis
            positive_words = set([
                'good', 'great', 'excellent', 'positive', 'best', 'love', 'happy', 'nice',
                'wonderful', 'fantastic', 'awesome', 'amazing', 'perfect', 'better', 'beautiful'
            ])
            
            negative_words = set([
                'bad', 'terrible', 'awful', 'negative', 'worst', 'hate', 'sad', 'poor',
                'horrible', 'disappointing', 'frustrating', 'annoying', 'worse', 'ugly', 'wrong'
            ])
            
            if st.button("Analyze Sentiment"):
                with st.spinner("Analyzing sentiment..."):
                    # Tokenize
                    words = re.findall(r'\w+', text_input.lower())
                    
                    # Count sentiment words
                    positive_count = sum(1 for word in words if word in positive_words)
                    negative_count = sum(1 for word in words if word in negative_words)
                    
                    # Calculate simple score
                    if positive_count + negative_count > 0:
                        sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                    else:
                        sentiment_score = 0
                    
                    # Determine sentiment label
                    if sentiment_score > 0.1:
                        sentiment = "Positive"
                        color = "green"
                    elif sentiment_score < -0.1:
                        sentiment = "Negative"
                        color = "red"
                    else:
                        sentiment = "Neutral"
                        color = "blue"
                    
                    # Display results
                    st.markdown(f"#### Sentiment: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
                    
                    # Display score
                    st.markdown(f"**Sentiment Score:** {sentiment_score:.2f} (-1.0 to 1.0)")
                    
                    # Display counts
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Positive Words", positive_count)
                    with col2:
                        st.metric("Negative Words", negative_count)
                    with col3:
                        neutral_count = len(words) - positive_count - negative_count
                        st.metric("Neutral Words", neutral_count)
                    
                    # Display as a gauge chart
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(['Sentiment'], [sentiment_score], color=color)
                    ax.set_xlim(-1, 1)
                    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    ax.set_title("Sentiment Score")
                    
                    # Add labels
                    ax.text(-0.9, 0, "Negative", fontsize=10, va='center')
                    ax.text(0.7, 0, "Positive", fontsize=10, va='center')
                    
                    st.pyplot(fig)
        else:
            st.info("Please enter or upload text in the Text Analysis tab first.")
    else:
        # Advanced sentiment analysis with transformers
        from transformers import pipeline
        
        st.markdown("#### Advanced Sentiment Analysis")
        
        if "text_input" in locals() and text_input:
            if st.button("Analyze Sentiment with Transformers"):
                with st.spinner("Analyzing sentiment using transformers model (this may take a moment)..."):
                    try:
                        # Initialize sentiment analysis pipeline
                        sentiment_analyzer = pipeline("sentiment-analysis")
                        
                        # For long texts, analyze by sentences and average the results
                        sentences = re.split(r'[.!?]+', text_input)
                        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
                        
                        if sentences:
                            # Use the first few sentences for efficiency
                            sample_size = min(10, len(sentences))
                            results = []
                            
                            for i, sentence in enumerate(sentences[:sample_size]):
                                result = sentiment_analyzer(sentence)[0]
                                results.append(result)
                                st.text(f"Analyzing sentence {i+1}/{sample_size}...")
                            
                            # Calculate overall sentiment
                            positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
                            negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
                            
                            if positive_count > negative_count:
                                overall_sentiment = "Positive"
                                color = "green"
                            elif negative_count > positive_count:
                                overall_sentiment = "Negative"
                                color = "red"
                            else:
                                overall_sentiment = "Neutral"
                                color = "blue"
                            
                            # Display results
                            st.markdown(f"#### Overall Sentiment: <span style='color:{color}'>{overall_sentiment}</span>", unsafe_allow_html=True)
                            
                            # Display sentence-level analysis
                            st.markdown("#### Sentence-level Analysis")
                            
                            for i, (sentence, result) in enumerate(zip(sentences[:sample_size], results)):
                                sentiment = result['label']
                                score = result['score']
                                
                                if sentiment == 'POSITIVE':
                                    emoji = "😊"
                                    color = "green"
                                else:
                                    emoji = "😟"
                                    color = "red"
                                
                                st.markdown(f"{emoji} **Sentence {i+1}** ({sentiment}, confidence: {score:.2f})")
                                st.markdown(f"> {sentence}")
                        else:
                            st.warning("No valid sentences found in the text.")
                    except Exception as e:
                        st.error(f"Error performing sentiment analysis: {str(e)}")
                        st.info("Falling back to basic sentiment analysis...")
                        
                        # Fallback to basic analysis
                        st.markdown("#### Basic Sentiment Analysis (Fallback)")
                        
                        # Simple word count approach
                        positive_words = set([
                            'good', 'great', 'excellent', 'positive', 'best', 'love', 'happy', 
                            'nice', 'wonderful', 'fantastic', 'awesome', 'amazing'
                        ])
                        
                        negative_words = set([
                            'bad', 'terrible', 'awful', 'negative', 'worst', 'hate', 'sad', 
                            'poor', 'horrible', 'disappointing', 'frustrating'
                        ])
                        
                        # Tokenize
                        words = re.findall(r'\w+', text_input.lower())
                        
                        # Count sentiment words
                        positive_count = sum(1 for word in words if word in positive_words)
                        negative_count = sum(1 for word in words if word in negative_words)
                        
                        # Calculate simple score
                        if positive_count + negative_count > 0:
                            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
                        else:
                            sentiment_score = 0
                        
                        # Determine sentiment label
                        if sentiment_score > 0.1:
                            sentiment = "Positive"
                        elif sentiment_score < -0.1:
                            sentiment = "Negative"
                        else:
                            sentiment = "Neutral"
                        
                        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
                        st.markdown(f"**Overall Sentiment:** {sentiment}")
        else:
            st.info("Please enter or upload text in the Text Analysis tab first.")

# Footer with development status
st.markdown("---")
st.markdown("**Note:** Some advanced features require additional Python packages. Install them for full functionality.") 