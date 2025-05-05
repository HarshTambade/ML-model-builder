"""
ALPHA - End-to-End Machine Learning Platform
Dashboard Module
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import random
from pathlib import Path

# Import utility modules
from utils.config import MODELS_DIR, get_available_models
from utils.models import load_model
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, display_code_block, add_vertical_space
)
from utils.imports import is_package_available, get_optional_package, logger, fix_dataframe_dtypes

# Optional imports
PLOTLY_AVAILABLE = is_package_available('plotly')
if PLOTLY_AVAILABLE:
    import plotly.express as px
    import plotly.graph_objects as go

# Helper function for creating visualizations with fallback to matplotlib if plotly is not available
def create_line_chart(df, x_col, y_col, title, x_label=None, y_label=None):
    """Create a line chart using plotly if available, otherwise matplotlib."""
    if PLOTLY_AVAILABLE:
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=title,
            markers=True
        )
        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)
        fig.update_layout(hovermode="x unified")
        return st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(df[x_col], df[y_col], marker='o')
        plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return st.pyplot(plt)

def create_bar_chart(df, x_col, y_col, title, x_label=None, y_label=None):
    """Create a bar chart using plotly if available, otherwise matplotlib."""
    if PLOTLY_AVAILABLE:
        fig = px.bar(
            df, 
            x=x_col, 
            y=y_col,
            title=title
        )
        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)
        return st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        plt.bar(df[x_col], df[y_col])
        plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        return st.pyplot(plt)

def create_multi_line_chart(df, x_col, y_cols, title, x_label=None, y_label=None):
    """Create a multi-line chart using plotly if available, otherwise matplotlib."""
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for col in y_cols:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode='lines+markers',
                name=col.upper()
            ))
        fig.update_layout(
            title=title,
            hovermode="x unified"
        )
        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)
        return st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        for col in y_cols:
            plt.plot(df[x_col], df[col], marker='o', label=col.upper())
        plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return st.pyplot(plt)

def create_histogram(df, col, title, x_label=None):
    """Create a histogram using plotly if available, otherwise matplotlib."""
    if PLOTLY_AVAILABLE:
        fig = px.histogram(
            df, 
            x=col,
            nbins=20,
            title=title
        )
        if x_label:
            fig.update_layout(xaxis_title=x_label)
        return st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        plt.hist(df[col], bins=20)
        plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return st.pyplot(plt)

def create_scatter_plot(df, x_col, y_col, title, size_col=None, x_label=None, y_label=None):
    """Create a scatter plot using plotly if available, otherwise matplotlib."""
    if PLOTLY_AVAILABLE:
        if size_col:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                size=size_col,
                hover_data=["date"],
                title=title
            )
        else:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                hover_data=["date"],
                title=title
            )
        if x_label:
            fig.update_layout(xaxis_title=x_label)
        if y_label:
            fig.update_layout(yaxis_title=y_label)
        return st.plotly_chart(fig, use_container_width=True)
    else:
        plt.figure(figsize=(10, 5))
        if size_col:
            sizes = df[size_col] / df[size_col].max() * 200
            plt.scatter(df[x_col], df[y_col], s=sizes)
        else:
            plt.scatter(df[x_col], df[y_col])
        plt.title(title)
        if x_label:
            plt.xlabel(x_label)
        if y_label:
            plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return st.pyplot(plt)

# Configure the page
set_page_config(title="Dashboard")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Model Dashboard",
    description="Monitor model performance and system metrics",
    icon="📊"
)

# Function to generate mock historical performance data for demonstration purposes
def generate_mock_performance_history(model_info, days=30):
    """Generate mock performance history for a model to demonstrate dashboard."""
    try:
        base_metrics = model_info["metadata"].get("performance", {}).get("metrics", {})
        
        if not base_metrics:
            return pd.DataFrame()
        
        # Generate dates for the last N days
        end_date = datetime.now()
        dates = [(end_date - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
        dates.reverse()  # Oldest to newest
        
        # Create variations of the metrics for each date
        data = []
        
        for date in dates:
            entry = {"date": date}
            
            # Add slightly varied metrics
            for metric, value in base_metrics.items():
                # Add some random variation to the metric, with a slight upward trend
                trend_factor = dates.index(date) / len(dates) * 0.1  # 10% improvement over time
                noise = random.uniform(-0.05, 0.05)  # Random noise
                adjusted_value = value * (1 + trend_factor + noise)
                
                # Ensure values stay in reasonable ranges
                if metric.lower() in ["accuracy", "precision", "recall", "f1"]:
                    adjusted_value = min(max(adjusted_value, 0.5), 1.0)
                
                entry[metric] = adjusted_value
                
            # Add prediction volume (increasing over time)
            base_volume = 100
            growth_factor = 1 + (dates.index(date) / len(dates))  # Growth over time
            entry["prediction_volume"] = int(base_volume * growth_factor + random.randint(-20, 20))
            
            # Add response time (ms)
            entry["avg_response_time"] = 100 + random.randint(-20, 50)
            
            # Add system load (percentage)
            entry["system_load"] = 20 + random.randint(0, 40)
            
            data.append(entry)
        
        # Convert to DataFrame and use the utility function to fix data types
        df = pd.DataFrame(data)
        return fix_dataframe_dtypes(df)
    except Exception as e:
        logger.error(f"Error generating mock data: {str(e)}")
        # Return empty DataFrame if there's an error
        return pd.DataFrame()

# Create dashboard tabs
dashboard_tabs = create_tab_panels(
    "Overview", "Performance Metrics", "System Metrics", "Model Comparison"
)

# Get available models
models = get_available_models()

if not models:
    st.warning("No trained models available for dashboard visualization. Please train a model in the Model Training module.")
    
    # Add a button to navigate to the Model Training page
    if st.button("Go to Model Training"):
        st.switch_page("pages/2_Model_Training.py")
else:
    # Model selection in sidebar
    with st.sidebar:
        st.markdown("## 📊 Dashboard Settings")
        model_names = [model["name"] for model in models]
        selected_model_name = st.selectbox("Select a model", model_names)
        
        # Find the selected model
        selected_model_info = next(
            (model for model in models if model["name"] == selected_model_name),
            None
        )
        
        # Time range selection
        st.markdown("### Time Range")
        time_range = st.radio(
            "Select time range",
            ["Last 7 days", "Last 30 days", "Last 90 days"]
        )
        
        if time_range == "Last 7 days":
            days = 7
        elif time_range == "Last 30 days":
            days = 30
        else:
            days = 90
        
        # Refresh button
        if st.button("Refresh Dashboard"):
            st.experimental_rerun()
    
    if selected_model_info:
        # Extract model ID from path
        model_path = Path(selected_model_info["path"])
        model_id = model_path.name
        
        # Load the model details
        model_info = load_model(model_id)
        
        if model_info:
            # Generate mock performance history for demonstration
            performance_history = generate_mock_performance_history(model_info, days)
            
            # Tab 1: Overview
            with dashboard_tabs[0]:
                st.markdown(f"### {model_info['metadata']['name']} Overview")
                
                # Display model metadata
                col1, col2, col3 = st.columns(3)
                col1.metric("Type", model_info["metadata"]["type"].capitalize())
                col2.metric("Created", model_info["metadata"]["created_at"])
                
                # Get a performance metric to display
                perf = model_info["metadata"].get("performance", {}).get("metrics", {})
                if perf:
                    metric_name = next(iter(perf.keys()))
                    metric_value = perf[metric_name]
                    
                    # Calculate trend if we have history
                    if not performance_history.empty:
                        last_value = performance_history[metric_name].iloc[-1]
                        delta = last_value - metric_value
                        col3.metric(metric_name.upper(), f"{last_value:.4f}", f"{delta:.4f}")
                    else:
                        col3.metric(metric_name.upper(), f"{metric_value:.4f}")
                
                # Summary cards for key metrics
                st.markdown("### Key Performance Indicators")
                
                if not performance_history.empty:
                    kpi_cols = st.columns(3)
                    
                    # Prediction volume
                    total_predictions = performance_history["prediction_volume"].sum()
                    avg_daily = performance_history["prediction_volume"].mean()
                    kpi_cols[0].metric(
                        "Total Predictions", 
                        f"{total_predictions:,}",
                        f"{avg_daily:.0f} avg/day"
                    )
                    
                    # Average response time
                    avg_time = performance_history["avg_response_time"].mean()
                    recent_avg = performance_history["avg_response_time"].iloc[-7:].mean()
                    time_delta = recent_avg - avg_time
                    kpi_cols[1].metric(
                        "Avg Response Time", 
                        f"{recent_avg:.1f} ms",
                        f"{time_delta:.1f} ms"
                    )
                    
                    # System utilization
                    avg_load = performance_history["system_load"].mean()
                    recent_load = performance_history["system_load"].iloc[-7:].mean()
                    load_delta = recent_load - avg_load
                    kpi_cols[2].metric(
                        "Avg System Load", 
                        f"{recent_load:.1f}%",
                        f"{load_delta:.1f}%"
                    )
                    
                    # Recent performance trend chart
                    st.markdown("### Recent Performance Trend")
                    
                    # Get the first metric for trend
                    metric_cols = [col for col in performance_history.columns 
                               if col not in ["date", "prediction_volume", "avg_response_time", "system_load"]]
                    if metric_cols:
                        main_metric = metric_cols[0]
                        create_line_chart(
                            performance_history, 
                            "date", 
                            main_metric,
                            f"{main_metric.upper()} Over Time",
                            "Date",
                            main_metric.upper()
                        )
                    
                    # Usage pattern over time
                    st.markdown("### Usage Pattern")
                    create_bar_chart(
                        performance_history, 
                        "date", 
                        "prediction_volume",
                        "Daily Prediction Volume",
                        "Date",
                        "Number of Predictions"
                    )
                
                else:
                    st.info("No historical performance data available for this model yet.")
            
            # Tab 2: Performance Metrics
            with dashboard_tabs[1]:
                st.markdown("### Model Performance Metrics")
                
                if not performance_history.empty:
                    # Select metrics to display
                    metric_cols = [col for col in performance_history.columns 
                                if col not in ["date", "prediction_volume", "avg_response_time", "system_load"]]
                    
                    if metric_cols:
                        # Multi-metric trend visualization
                        st.markdown("#### Performance Metrics Over Time")
                        
                        selected_metrics = st.multiselect(
                            "Select metrics to display",
                            options=metric_cols,
                            default=metric_cols[:2] if len(metric_cols) > 1 else metric_cols
                        )
                        
                        if selected_metrics:
                            create_multi_line_chart(
                                performance_history,
                                "date",
                                selected_metrics,
                                "Performance Metrics Trend",
                                "Date",
                                "Metric Value"
                            )
                            
                            # Detailed metrics table
                            st.markdown("#### Detailed Metrics")
                            metrics_df = performance_history[["date"] + selected_metrics]
                            st.dataframe(metrics_df.set_index("date"))
                            
                            # Download metrics data
                            csv = metrics_df.to_csv(index=False)
                            st.download_button(
                                label="Download Metrics Data",
                                data=csv,
                                file_name=f"{selected_model_name}_metrics.csv",
                                mime="text/csv"
                            )
                        
                        # Performance distribution
                        st.markdown("#### Metric Distribution")
                        
                        if metric_cols:
                            metric_for_dist = st.selectbox(
                                "Select metric for distribution analysis",
                                options=metric_cols
                            )
                            
                            create_histogram(
                                performance_history, 
                                metric_for_dist,
                                f"Distribution of {metric_for_dist.upper()}",
                                metric_for_dist.upper()
                            )
                            
                            # Basic statistics
                            st.markdown("#### Summary Statistics")
                            stats_df = performance_history[metric_cols].describe().T
                            st.dataframe(stats_df)
                            
                            # Add download button for the metrics data
                            csv = performance_history[["date"] + metric_cols].to_csv(index=False)
                            st.download_button(
                                label="Download Metrics Data",
                                data=csv,
                                file_name=f"{selected_model_name}_metrics.csv",
                                mime="text/csv"
                            )
                    else:
                        st.info("No performance metrics available for this model.")
                else:
                    st.info("No historical performance data available for this model yet.")
            
            # Tab 3: System Metrics
            with dashboard_tabs[2]:
                st.markdown("### System Performance Metrics")
                
                if not performance_history.empty:
                    # Response time over time
                    st.markdown("#### Response Time Trend")
                    create_line_chart(
                        performance_history, 
                        "date", 
                        "avg_response_time",
                        "Average Response Time",
                        "Date",
                        "Response Time (ms)"
                    )
                    
                    # System load and prediction volume relation
                    st.markdown("#### System Load vs. Prediction Volume")
                    create_scatter_plot(
                        performance_history, 
                        "prediction_volume", 
                        "system_load",
                        "System Load vs. Prediction Volume",
                        "avg_response_time",
                        "Number of Predictions",
                        "System Load (%)"
                    )
                    
                    # System metrics over time
                    st.markdown("#### System Metrics Over Time")
                    system_metrics = st.multiselect(
                        "Select system metrics to display",
                        options=["system_load", "avg_response_time", "prediction_volume"],
                        default=["system_load", "avg_response_time"]
                    )
                    
                    if system_metrics:
                        # Create a normalized view for comparison
                        norm_df = performance_history.copy()
                        for metric in system_metrics:
                            norm_df[f"{metric}_norm"] = (norm_df[metric] - norm_df[metric].min()) / (norm_df[metric].max() - norm_df[metric].min())
                        
                        create_multi_line_chart(
                            norm_df,
                            "date",
                            [f"{metric}_norm" for metric in system_metrics],
                            "Normalized System Metrics Comparison",
                            "Date",
                            "Normalized Value (0-1)"
                        )
                        
                        # Detailed System Metrics Table
                        st.markdown("#### Detailed System Metrics")
                        sys_df = performance_history[["date"] + system_metrics]
                        st.dataframe(sys_df.set_index("date"))
                        
                        # Add download button for system metrics
                        sys_csv = sys_df.to_csv(index=False)
                        st.download_button(
                            label="Download System Metrics",
                            data=sys_csv,
                            file_name=f"{selected_model_name}_system_metrics.csv",
                            mime="text/csv"
                        )
                else:
                    st.info("No system metrics data available for this model yet.")
            
            # Tab 4: Model Comparison
            with dashboard_tabs[3]:
                st.markdown("### Model Comparison")
                
                if len(models) > 1:
                    # Select models to compare
                    model_options = {model["name"]: model["id"] for model in models}
                    compare_models = st.multiselect(
                        "Select models to compare",
                        options=list(model_options.keys()),
                        default=[selected_model_name]
                    )
                    
                    if len(compare_models) > 1:
                        # Prepare data for each model
                        comparison_data = []
                        for model_name in compare_models:
                            model_id = model_options[model_name]
                            compare_info = load_model(model_id)
                            
                            if compare_info:
                                # Get performance metrics
                                metrics = compare_info["metadata"].get("performance", {}).get("metrics", {})
                                if metrics:
                                    metrics_data = {
                                        "model_name": model_name,
                                        "model_type": compare_info["metadata"]["type"],
                                        "created_at": compare_info["metadata"]["created_at"]
                                    }
                                    metrics_data.update(metrics)
                                    comparison_data.append(metrics_data)
                        
                        if comparison_data:
                            # Create comparison dataframe
                            compare_df = pd.DataFrame(comparison_data)
                            
                            # Display comparison table
                            st.markdown("#### Model Performance Comparison")
                            st.dataframe(compare_df.set_index("model_name"))
                            
                            # Add download button for comparison data
                            comp_csv = compare_df.to_csv(index=False)
                            st.download_button(
                                label="Download Comparison Data",
                                data=comp_csv,
                                file_name="model_comparison.csv",
                                mime="text/csv"
                            )
                            
                            # Bar chart comparison
                            st.markdown("#### Visual Comparison")
                            compare_df_melted = compare_df.reset_index().melt(
                                id_vars=["model_name"], 
                                var_name="Metric", 
                                value_name="Value",
                                value_vars=[col for col in compare_df.columns if col not in ["model_name", "model_type", "created_at", "index"]]
                            )
                            
                            # Use our helper function for the bar chart
                            if not compare_df_melted.empty:
                                for metric in compare_df_melted["Metric"].unique():
                                    metric_df = compare_df_melted[compare_df_melted["Metric"] == metric]
                                    create_bar_chart(
                                        metric_df, 
                                        "model_name", 
                                        "Value", 
                                        f"{metric.upper()} Comparison",
                                        "Model",
                                        metric.upper()
                                    )
                            else:
                                st.info("No comparable metrics available for the selected models.")
                    else:
                        st.info("Please select at least two models for comparison.")
                else:
                    st.info("You need at least two trained models to use the comparison feature.")
        else:
            st.error("Error loading model. The model might be corrupted.")
            
# Dashboard footer with tips
st.markdown("---")
display_info_box(
    text="**Dashboard Tips:**\n- Monitor Regularly: Check your model performance at least weekly\n- Look for Trends: Sudden changes in metrics may indicate issues\n- Compare Versions: When deploying new models, compare with previous versions\n- System Load: High system load with slow response times may indicate scaling issues",
    type="info"
) 