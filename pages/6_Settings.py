"""
ALPHA - End-to-End Machine Learning Platform
Settings Module
"""

import streamlit as st
import os
import json
import yaml
import shutil
from datetime import datetime

# Import utility modules
from utils.config import (
    APP_CONFIG, DATASETS_DIR, MODELS_DIR,
    save_settings, load_settings, get_available_models, 
    get_available_datasets
)
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, add_vertical_space
)

# Configure the page
set_page_config(title="Settings")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="Settings",
    description="Configure your ALPHA ML Platform",
    icon="⚙️"
)

# Load current settings
current_settings = load_settings()

# Create settings tabs
settings_tabs = create_tab_panels(
    "General", "Storage", "Advanced", "Backup & Restore"
)

# Tab 1: General Settings
with settings_tabs[0]:
    st.markdown("### General Settings")
    
    # Theme settings
    st.markdown("#### Theme Settings")
    theme = st.selectbox(
        "Select theme",
        ["Light", "Dark", "Auto"],
        index=["Light", "Dark", "Auto"].index(current_settings.get("theme", "Auto"))
    )
    
    # UI settings
    st.markdown("#### UI Settings")
    sidebar_collapsed = st.checkbox(
        "Collapse sidebar by default",
        value=current_settings.get("sidebar_collapsed", False)
    )
    
    wide_mode = st.checkbox(
        "Use wide mode by default",
        value=current_settings.get("wide_mode", True)
    )
    
    show_tooltips = st.checkbox(
        "Show tooltips",
        value=current_settings.get("show_tooltips", True)
    )
    
    # Notification settings
    st.markdown("#### Notification Settings")
    notifications_enabled = st.checkbox(
        "Enable notifications",
        value=current_settings.get("notifications_enabled", True)
    )
    
    notification_level = st.select_slider(
        "Notification level",
        options=["None", "Critical", "Errors", "Warnings", "Info", "All"],
        value=current_settings.get("notification_level", "Warnings")
    )

# Tab 2: Storage Settings
with settings_tabs[1]:
    st.markdown("### Storage Settings")
    
    # Storage paths
    st.markdown("#### Storage Paths")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Current Paths")
        st.info(f"Data Directory: `{DATASETS_DIR}`")
        st.info(f"Models Directory: `{MODELS_DIR}`")
        
        # Display storage usage
        st.markdown("##### Storage Usage")
        
        # Get storage stats
        def get_dir_size(path):
            total_size = 0
            if os.path.exists(path):
                for dirpath, dirnames, filenames in os.walk(path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)
            return total_size
        
        data_size = get_dir_size(DATASETS_DIR)
        models_size = get_dir_size(MODELS_DIR)
        total_size = data_size + models_size
        
        # Convert to appropriate units
        def format_size(size_bytes):
            if size_bytes == 0:
                return "0 B"
            size_name = ("B", "KB", "MB", "GB", "TB")
            i = 0
            while size_bytes >= 1024 and i < len(size_name)-1:
                size_bytes /= 1024
                i += 1
            return f"{size_bytes:.2f} {size_name[i]}"
        
        st.metric("Data Storage", format_size(data_size))
        st.metric("Models Storage", format_size(models_size))
        st.metric("Total Storage", format_size(total_size))
    
    with col2:
        st.markdown("##### Cleanup Options")
        
        # Cleanup datasets
        st.markdown("**Dataset Cleanup**")
        available_datasets = get_available_datasets()
        if available_datasets:
            datasets_to_delete = st.multiselect(
                "Select datasets to delete",
                options=[ds["name"] for ds in available_datasets]
            )
            
            if datasets_to_delete and st.button("Delete Selected Datasets"):
                for ds_name in datasets_to_delete:
                    # Find dataset path and delete
                    dataset = next((ds for ds in available_datasets if ds["name"] == ds_name), None)
                    if dataset and os.path.exists(dataset["path"]):
                        try:
                            shutil.rmtree(dataset["path"])
                            st.success(f"Deleted dataset: {ds_name}")
                        except Exception as e:
                            st.error(f"Error deleting dataset {ds_name}: {str(e)}")
        else:
            st.info("No datasets available for cleanup")
        
        # Cleanup models
        st.markdown("**Model Cleanup**")
        available_models = get_available_models()
        if available_models:
            models_to_delete = st.multiselect(
                "Select models to delete",
                options=[model["name"] for model in available_models]
            )
            
            if models_to_delete and st.button("Delete Selected Models"):
                for model_name in models_to_delete:
                    # Find model path and delete
                    model = next((m for m in available_models if m["name"] == model_name), None)
                    if model and os.path.exists(model["path"]):
                        try:
                            shutil.rmtree(model["path"])
                            st.success(f"Deleted model: {model_name}")
                        except Exception as e:
                            st.error(f"Error deleting model {model_name}: {str(e)}")
        else:
            st.info("No models available for cleanup")

# Tab 3: Advanced Settings
with settings_tabs[2]:
    st.markdown("### Advanced Settings")
    
    # Performance settings
    st.markdown("#### Performance Settings")
    
    cache_enabled = st.checkbox(
        "Enable caching",
        value=current_settings.get("cache_enabled", True),
        help="Cache data and model predictions to improve performance"
    )
    
    parallelism = st.slider(
        "Parallelism level",
        min_value=1,
        max_value=8,
        value=current_settings.get("parallelism", 2),
        help="Number of parallel processes for computation (higher uses more CPU)"
    )
    
    # Logging settings
    st.markdown("#### Logging Settings")
    
    log_level = st.select_slider(
        "Logging level",
        options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        value=current_settings.get("log_level", "INFO"),
        help="Set logging detail level"
    )
    
    # Developer settings
    st.markdown("#### Developer Settings")
    
    with st.expander("Developer Options"):
        developer_mode = st.checkbox(
            "Enable developer mode",
            value=current_settings.get("developer_mode", False),
            help="Show advanced options and debug information"
        )
        
        allow_custom_code = st.checkbox(
            "Allow custom code execution",
            value=current_settings.get("allow_custom_code", False),
            help="Enable execution of custom code (CAUTION: security risk)"
        )
        
        if allow_custom_code:
            st.warning("⚠️ Enabling custom code execution poses security risks. Use with caution!")
            
        experimental_features = st.checkbox(
            "Enable experimental features",
            value=current_settings.get("experimental_features", False),
            help="Access experimental features that may not be stable"
        )
        
        if experimental_features:
            st.info("ℹ️ Experimental features enabled. Some functionality may be unstable.")

# Tab 4: Backup & Restore
with settings_tabs[3]:
    st.markdown("### Backup & Restore")
    
    # Backup settings
    st.markdown("#### Create Backup")
    
    backup_options = st.multiselect(
        "Select what to include in backup",
        options=["Settings", "Datasets", "Models"],
        default=["Settings", "Datasets", "Models"]
    )
    
    if st.button("Create Backup"):
        if backup_options:
            # Create backup data
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "version": APP_CONFIG["version"],
                "content": {}
            }
            
            if "Settings" in backup_options:
                backup_data["content"]["settings"] = current_settings
            
            if "Datasets" in backup_options:
                datasets = get_available_datasets()
                backup_data["content"]["datasets"] = [{"name": ds["name"], "metadata": ds["metadata"]} for ds in datasets]
            
            if "Models" in backup_options:
                models = get_available_models()
                backup_data["content"]["models"] = [{"name": model["name"], "metadata": model["metadata"]} for model in models]
            
            # Convert to JSON
            backup_json = json.dumps(backup_data, indent=2)
            
            # Create downloadable file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Backup File",
                data=backup_json,
                file_name=f"alpha_backup_{timestamp}.json",
                mime="application/json"
            )
            
            st.success("Backup created successfully!")
        else:
            st.error("Please select at least one option to include in the backup")
    
    # Restore settings
    st.markdown("#### Restore from Backup")
    
    uploaded_file = st.file_uploader("Upload backup file", type=["json"])
    
    if uploaded_file is not None:
        try:
            # Read backup file
            backup_data = json.load(uploaded_file)
            
            # Show backup information
            st.info(f"Backup created on: {backup_data.get('timestamp', 'Unknown')}")
            st.info(f"Backup version: {backup_data.get('version', 'Unknown')}")
            
            # Backup content
            content = backup_data.get("content", {})
            restore_options = []
            
            if "settings" in content:
                restore_options.append("Settings")
            
            if "datasets" in content:
                restore_options.append(f"Datasets ({len(content['datasets'])})")
            
            if "models" in content:
                restore_options.append(f"Models ({len(content['models'])})")
            
            # Restore options
            selected_restore = st.multiselect(
                "Select what to restore",
                options=restore_options,
                default=restore_options
            )
            
            if selected_restore and st.button("Restore Selected"):
                if "Settings" in selected_restore:
                    # Restore settings
                    new_settings = content.get("settings", {})
                    current_settings.update(new_settings)
                    save_settings(current_settings)
                    st.success("Settings restored successfully!")
                
                # Note for datasets and models
                if any(opt.startswith("Datasets") for opt in selected_restore) or any(opt.startswith("Models") for opt in selected_restore):
                    st.info("Datasets and models metadata restored. Note that actual data files need to be imported separately.")
                
                st.success("Restoration completed!")
        except Exception as e:
            st.error(f"Error restoring from backup: {str(e)}")

# Save changes button
st.markdown("---")
if st.button("Save Settings", type="primary"):
    # Update settings
    new_settings = {
        "theme": theme,
        "sidebar_collapsed": sidebar_collapsed,
        "wide_mode": wide_mode,
        "show_tooltips": show_tooltips,
        "notifications_enabled": notifications_enabled,
        "notification_level": notification_level,
        "cache_enabled": cache_enabled,
        "parallelism": parallelism,
        "log_level": log_level,
        "developer_mode": developer_mode if 'developer_mode' in locals() else current_settings.get("developer_mode", False),
        "allow_custom_code": allow_custom_code if 'allow_custom_code' in locals() else current_settings.get("allow_custom_code", False),
        "experimental_features": experimental_features if 'experimental_features' in locals() else current_settings.get("experimental_features", False),
    }
    
    # Save settings
    save_settings(new_settings)
    st.success("Settings saved successfully!")

# Display information about settings
with st.sidebar:
    st.markdown("## ⚙️ Settings Help")
    
    with st.expander("💡 Settings Tips"):
        st.markdown("""
        ### Settings Tips
        
        1. **Theme**: Choose a theme that works best for your environment
        
        2. **Storage**: Regularly clean up unused datasets and models
        
        3. **Backup**: Create periodic backups of your settings and metadata
        
        4. **Performance**: Adjust parallelism based on your hardware capabilities
        
        5. **Developer Mode**: Only enable if you need advanced customization
        """) 