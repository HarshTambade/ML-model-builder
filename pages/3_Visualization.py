"""
ALPHA - End-to-End Machine Learning Platform
Visualization Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

# Import utility modules
from utils.config import get_available_datasets
from utils.data import get_dataset_by_id
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)
from utils.imports import is_package_available, logger, fix_dataframe_dtypes, validate_dataframe_for_streamlit

# Configure the page
set_page_config(title="Visualization")

# Display sidebar navigation
sidebar_navigation()

# Dependency checks
if not is_package_available('pandas'):
    st.error('Pandas is required for visualization. Please install pandas.')
    st.stop()
if not is_package_available('matplotlib'):
    st.warning('Matplotlib is not available. Some visualizations may not work.')
if not is_package_available('seaborn'):
    st.warning('Seaborn is not available. Some visualizations may not work.')
if not is_package_available('plotly'):
    st.warning('Plotly is not available. Interactive visualizations may not work.')
if not is_package_available('altair'):
    st.warning('Altair is not available. Some visualizations may not work.')

# Main content
page_header(
    title="Visualization",
    description="Create interactive visualizations of your data",
    icon="📈"
)

# Function to create visualizations based on type
def create_visualization(df, viz_type, settings):
    """Create a visualization based on the type and settings."""
    if viz_type == "Scatter Plot":
        x_col = settings.get("x_column")
        y_col = settings.get("y_column")
        color_col = settings.get("color_column")
        
        fig = px.scatter(
            df, 
            x=x_col, 
            y=y_col,
            color=color_col if color_col != "None" else None,
            title=f"{y_col} vs {x_col}",
            labels={x_col: x_col, y_col: y_col},
            hover_data=df.columns,
        )
        
        return fig
    
    elif viz_type == "Line Chart":
        x_col = settings.get("x_column")
        y_col = settings.get("y_column")
        group_col = settings.get("group_column")
        
        if group_col != "None":
            fig = px.line(
                df, 
                x=x_col, 
                y=y_col,
                color=group_col,
                title=f"{y_col} over {x_col} by {group_col}",
                labels={x_col: x_col, y_col: y_col},
            )
        else:
            fig = px.line(
                df, 
                x=x_col, 
                y=y_col,
                title=f"{y_col} over {x_col}",
                labels={x_col: x_col, y_col: y_col},
            )
        
        return fig
    
    elif viz_type == "Bar Chart":
        x_col = settings.get("x_column")
        y_col = settings.get("y_column")
        group_col = settings.get("group_column")
        
        if group_col != "None":
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col,
                color=group_col,
                title=f"{y_col} by {x_col}",
                labels={x_col: x_col, y_col: y_col},
                barmode="group" if settings.get("bar_mode") == "Group" else "stack",
            )
        else:
            fig = px.bar(
                df, 
                x=x_col, 
                y=y_col,
                title=f"{y_col} by {x_col}",
                labels={x_col: x_col, y_col: y_col},
            )
        
        return fig
    
    elif viz_type == "Histogram":
        col = settings.get("column")
        bins = settings.get("bins")
        
        fig = px.histogram(
            df, 
            x=col,
            nbins=bins,
            title=f"Distribution of {col}",
            labels={col: col},
        )
        
        return fig
    
    elif viz_type == "Box Plot":
        x_col = settings.get("x_column")
        y_col = settings.get("y_column")
        
        fig = px.box(
            df, 
            x=x_col, 
            y=y_col,
            title=f"Box Plot of {y_col} by {x_col}",
            labels={x_col: x_col, y_col: y_col},
        )
        
        return fig
    
    elif viz_type == "Heatmap":
        # Get correlation matrix
        corr_df = df.select_dtypes(include=["int64", "float64"]).corr()
        
        fig = px.imshow(
            corr_df,
            text_auto=True,
            title="Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        
        return fig
    
    elif viz_type == "Pie Chart":
        col = settings.get("column")
        
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]
        
        fig = px.pie(
            value_counts, 
            values="count", 
            names=col,
            title=f"Distribution of {col}",
        )
        
        return fig
    
    elif viz_type == "3D Scatter":
        x_col = settings.get("x_column")
        y_col = settings.get("y_column")
        z_col = settings.get("z_column")
        color_col = settings.get("color_column")
        
        fig = px.scatter_3d(
            df, 
            x=x_col, 
            y=y_col,
            z=z_col,
            color=color_col if color_col != "None" else None,
            title=f"3D Scatter Plot",
            labels={x_col: x_col, y_col: y_col, z_col: z_col},
        )
        
        return fig
    
    elif viz_type == "Dimensionality Reduction":
        method = settings.get("method")
        color_col = settings.get("color_column")
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        
        if len(numeric_df.columns) < 2:
            st.error("Not enough numeric columns for dimensionality reduction.")
            return None
        
        # Apply dimensionality reduction
        if method == "PCA":
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(numeric_df)
            explained_var = reducer.explained_variance_ratio_
            subtitle = f"Explained variance: {explained_var[0]:.2f}, {explained_var[1]:.2f}"
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(numeric_df)
            subtitle = "t-SNE projection"
        
        # Create dataframe with reduced dimensions
        reduced_df = pd.DataFrame({
            "Component 1": reduced_data[:, 0],
            "Component 2": reduced_data[:, 1],
        })
        
        # Add color column if specified
        if color_col != "None":
            reduced_df[color_col] = df[color_col].values
        
        # Create plot
        fig = px.scatter(
            reduced_df, 
            x="Component 1", 
            y="Component 2",
            color=color_col if color_col != "None" else None,
            title=f"{method} Visualization - {subtitle}",
        )
        
        return fig
    
    else:
        st.warning(f"Visualization type '{viz_type}' is not implemented yet.")
        return None

# Main visualization interface
st.markdown("### Create Your Visualization")

# Get available datasets
datasets = get_available_datasets()

if not datasets:
    st.warning("No datasets available. Please upload or generate a dataset in the Data Management module.")
else:
    # Dataset selection
    dataset_names = [dataset["name"] for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)
    
    # Find the selected dataset
    selected_dataset = next(
        (dataset for dataset in datasets if dataset["name"] == selected_dataset_name),
        None
    )
    
    if selected_dataset:
        # Extract dataset path to get the ID
        dataset_path = Path(selected_dataset["path"])
        dataset_id = dataset_path.name
        
        # Get the dataset
        dataset_info = get_dataset_by_id(dataset_id)
        
        if dataset_info and dataset_info["data"] is not None:
            df = dataset_info["data"]
            
            # Validate DataFrame before display
            is_valid, msg = validate_dataframe_for_streamlit(df)
            if not is_valid:
                st.error(f"Cannot display DataFrame: {msg}")
            else:
                # Display dataset preview
                with st.expander("Dataset Preview", expanded=False):
                    st.dataframe(df.head())
                
                # Visualization selection
                visualization_types = [
                    "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", 
                    "Box Plot", "Heatmap", "Pie Chart", "3D Scatter",
                    "Dimensionality Reduction"
                ]
                
                selected_viz = st.selectbox("Select Visualization Type", visualization_types)
                
                # Visualization settings based on type
                settings = {}
                
                # Columns selector
                numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
                categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
                all_cols = df.columns.tolist()
                
                st.markdown("#### Configure Visualization")
                
                if selected_viz == "Scatter Plot":
                    col1, col2 = st.columns(2)
                    settings["x_column"] = col1.selectbox("X-axis", numeric_cols, key="scatter_x")
                    settings["y_column"] = col2.selectbox("Y-axis", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
                    settings["color_column"] = st.selectbox("Color by", ["None"] + all_cols, key="scatter_color")
                
                elif selected_viz == "Line Chart":
                    col1, col2 = st.columns(2)
                    settings["x_column"] = col1.selectbox("X-axis", numeric_cols, key="line_x")
                    settings["y_column"] = col2.selectbox("Y-axis", numeric_cols, key="line_y", index=min(1, len(numeric_cols)-1))
                    settings["group_column"] = st.selectbox("Group by", ["None"] + categorical_cols, key="line_group")
                
                elif selected_viz == "Bar Chart":
                    col1, col2 = st.columns(2)
                    settings["x_column"] = col1.selectbox("X-axis", all_cols, key="bar_x")
                    settings["y_column"] = col2.selectbox("Y-axis", numeric_cols, key="bar_y")
                    settings["group_column"] = st.selectbox("Group by", ["None"] + categorical_cols, key="bar_group")
                    settings["bar_mode"] = st.radio("Bar Mode", ["Group", "Stack"], horizontal=True)
                
                elif selected_viz == "Histogram":
                    col1, col2 = st.columns(2)
                    settings["column"] = col1.selectbox("Column", numeric_cols, key="hist_col")
                    settings["bins"] = col2.slider("Number of Bins", 5, 100, 20)
                
                elif selected_viz == "Box Plot":
                    col1, col2 = st.columns(2)
                    settings["x_column"] = col1.selectbox("X-axis (Category)", categorical_cols if categorical_cols else all_cols, key="box_x")
                    settings["y_column"] = col2.selectbox("Y-axis (Value)", numeric_cols, key="box_y")
                
                elif selected_viz == "Heatmap":
                    st.info("Heatmap will show correlations between all numeric columns.")
                
                elif selected_viz == "Pie Chart":
                    settings["column"] = st.selectbox("Column", categorical_cols if categorical_cols else all_cols, key="pie_col")
                
                elif selected_viz == "3D Scatter":
                    col1, col2, col3 = st.columns(3)
                    settings["x_column"] = col1.selectbox("X-axis", numeric_cols, key="3d_x")
                    settings["y_column"] = col2.selectbox("Y-axis", numeric_cols, key="3d_y", index=min(1, len(numeric_cols)-1))
                    z_index = min(2, len(numeric_cols)-1) if len(numeric_cols) > 2 else 0
                    settings["z_column"] = col3.selectbox("Z-axis", numeric_cols, key="3d_z", index=z_index)
                    settings["color_column"] = st.selectbox("Color by", ["None"] + all_cols, key="3d_color")
                
                elif selected_viz == "Dimensionality Reduction":
                    col1, col2 = st.columns(2)
                    settings["method"] = col1.selectbox("Method", ["PCA", "t-SNE"], key="dr_method")
                    settings["color_column"] = col2.selectbox("Color by", ["None"] + all_cols, key="dr_color")
                
                # Generate visualization
                if st.button("Generate Visualization"):
                    with st.spinner("Creating visualization..."):
                        fig = create_visualization(df, selected_viz, settings)
                        
                        if fig is not None:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Export options
                            st.markdown("#### Export Options")
                            col1, col2 = st.columns(2)
                            
                            # HTML export
                            html_bytes = fig.to_html(include_plotlyjs="cdn").encode()
                            col1.download_button(
                                label="Download as HTML",
                                data=html_bytes,
                                file_name=f"{selected_dataset_name}_{selected_viz.replace(' ', '_')}.html",
                                mime="text/html"
                            )
                            
                            # PNG export
                            img_bytes = fig.to_image(format="png", width=1200, height=800)
                            col2.download_button(
                                label="Download as PNG",
                                data=img_bytes,
                                file_name=f"{selected_dataset_name}_{selected_viz.replace(' ', '_')}.png",
                                mime="image/png"
                            )
        else:
            st.error("Error loading dataset data. The dataset might be corrupted.")

# Dashboard creation section
st.markdown("---")
st.markdown("### Custom Dashboard Creator")
st.info("This feature allows you to create custom dashboards with multiple visualizations. Coming soon!")

# Additional sidebar options
with st.sidebar:
    st.markdown("## 📊 Visualization Options")
    
    with st.expander("💡 Visualization Tips"):
        st.markdown("""
        ### Tips for Effective Visualizations
        
        1. **Choose the Right Chart Type**: Use bar charts for comparing categories, line charts for trends over time, scatter plots for relationships, etc.
        
        2. **Color Usage**: Use colors strategically to highlight important data or to represent categories
        
        3. **Simplicity**: Keep visualizations clean and avoid chart junk or unnecessary elements
        
        4. **Data Labels**: Include clear labels to make your visualization self-explanatory
        
        5. **Interactive Elements**: Use Plotly's interactive features to let users explore the data
        """) 