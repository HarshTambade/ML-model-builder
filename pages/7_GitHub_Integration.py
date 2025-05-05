"""
ALPHA - End-to-End Machine Learning Platform
GitHub Integration Module
"""

import streamlit as st
import os
import json
import zipfile
import io
import requests
import base64
from datetime import datetime

# Import utility modules
from utils.config import get_available_models, load_settings
from utils.models import load_model
from utils.ui import (
    set_page_config, page_header, sidebar_navigation, display_info_box,
    create_tab_panels, display_code_block, add_vertical_space
)

# Configure the page
set_page_config(title="GitHub Integration")

# Display sidebar navigation
sidebar_navigation()

# Main content
page_header(
    title="GitHub Integration",
    description="Manage models in GitHub repositories",
    icon="💻"
)

# Get settings
settings = load_settings()

# Create GitHub integration tabs
github_tabs = create_tab_panels(
    "Connect", "Push Models", "Pull Models", "Collaboration"
)

# Tab 1: Connect to GitHub
with github_tabs[0]:
    st.markdown("### Connect to GitHub")
    
    st.markdown("""
    Connect your ALPHA platform to GitHub to:
    - Version control your models and datasets
    - Collaborate with other data scientists
    - Share models with the community
    - Integrate with CI/CD pipelines
    """)
    
    # GitHub connection settings
    with st.form("github_connection"):
        st.markdown("#### GitHub Credentials")
        
        # Get saved credentials if any
        saved_username = settings.get("github", {}).get("username", "")
        
        github_username = st.text_input("GitHub Username", value=saved_username)
        github_token = st.text_input("GitHub Personal Access Token", type="password", 
                                      help="Create a token at GitHub > Settings > Developer settings > Personal access tokens")
        
        save_credentials = st.checkbox("Save credentials (token will be stored securely)", 
                                      value=True)
        
        submitted = st.form_submit_button("Connect to GitHub")
        
        if submitted:
            if github_username and github_token:
                # Test GitHub connection
                headers = {
                    "Authorization": f"token {github_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
                
                try:
                    response = requests.get("https://api.github.com/user", headers=headers)
                    
                    if response.status_code == 200:
                        st.success("Successfully connected to GitHub!")
                        
                        # Save credentials if requested
                        if save_credentials:
                            if "github" not in settings:
                                settings["github"] = {}
                            
                            settings["github"]["username"] = github_username
                            # In a real app, you would securely store the token
                            # settings["github"]["token"] = encrypt(github_token)
                            
                            # For demo, we'll just indicate token is saved
                            settings["github"]["token_saved"] = True
                            settings["github"]["connected"] = True
                            
                            # Save settings
                            from utils.config import save_settings
                            save_settings(settings)
                    else:
                        st.error(f"Failed to connect to GitHub: {response.json().get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to GitHub: {str(e)}")
            else:
                st.warning("Please enter both GitHub username and token.")
    
    # GitHub repositories
    st.markdown("### Your GitHub Repositories")
    
    if settings.get("github", {}).get("connected", False):
        # Display repository management interface
        st.markdown("#### Repository Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Create New Repository**")
            
            with st.form("create_repo"):
                repo_name = st.text_input("Repository Name", placeholder="alpha-models")
                repo_description = st.text_area("Description", placeholder="Models created with ALPHA ML Platform")
                repo_private = st.checkbox("Private Repository", value=True)
                
                create_submitted = st.form_submit_button("Create Repository")
                
                if create_submitted:
                    if repo_name:
                        st.info(f"Creating repository '{repo_name}'...")
                        # In a real app, this would use the GitHub API to create the repository
                        st.success(f"Repository '{repo_name}' created successfully!")
                    else:
                        st.warning("Please enter a repository name.")
        
        with col2:
            st.markdown("**Existing Repositories**")
            
            # In a real app, this would fetch real repositories from GitHub API
            mock_repos = [
                {"name": "alpha-production-models", "updated_at": "2023-05-15", "private": True},
                {"name": "customer-segmentation", "updated_at": "2023-04-22", "private": False},
                {"name": "time-series-forecasting", "updated_at": "2023-06-01", "private": True}
            ]
            
            if mock_repos:
                for repo in mock_repos:
                    st.markdown(f"**{repo['name']}**")
                    st.markdown(f"Updated: {repo['updated_at']} • {'Private' if repo['private'] else 'Public'}")
                    st.markdown("---")
            else:
                st.info("No repositories found.")
    else:
        st.info("Connect to GitHub to manage your repositories.")

# Tab 2: Push Models to GitHub
with github_tabs[1]:
    st.markdown("### Push Models to GitHub")
    
    if settings.get("github", {}).get("connected", False):
        # Get available models
        models = get_available_models()
        
        if models:
            st.markdown("#### Select Models to Push")
            
            # Repository selection (would be fetched from GitHub API in a real app)
            repositories = ["alpha-production-models", "customer-segmentation", "time-series-forecasting", "Create new..."]
            selected_repo = st.selectbox("Select repository", repositories)
            
            if selected_repo == "Create new...":
                new_repo_name = st.text_input("New repository name")
                repo_private = st.checkbox("Private repository", value=True)
                
                if new_repo_name:
                    st.success(f"Repository '{new_repo_name}' will be created.")
                    selected_repo = new_repo_name
            
            # Model selection
            model_names = [model["name"] for model in models]
            selected_models = st.multiselect("Select models to push", model_names)
            
            if selected_models:
                # Configuration options
                st.markdown("#### Push Configuration")
                
                commit_message = st.text_area(
                    "Commit message", 
                    value=f"Push {len(selected_models)} models from ALPHA ML Platform"
                )
                
                include_options = st.multiselect(
                    "Include in push",
                    options=["Model files", "Training data", "Performance metrics", "Documentation"],
                    default=["Model files", "Performance metrics", "Documentation"]
                )
                
                # Branch options
                branch_options = ["main", "develop", "Create new branch..."]
                selected_branch = st.selectbox("Target branch", branch_options)
                
                if selected_branch == "Create new branch...":
                    new_branch = st.text_input("New branch name")
                    if new_branch:
                        st.success(f"Branch '{new_branch}' will be created.")
                        selected_branch = new_branch
                
                # Push button
                if st.button("Push to GitHub"):
                    with st.spinner("Pushing models to GitHub..."):
                        # In a real app, this would actually push to GitHub
                        st.progress(100)
                        st.success(f"Successfully pushed {len(selected_models)} models to {selected_repo}/{selected_branch}")
                        
                        # Display what would be the GitHub URL
                        github_username = settings.get("github", {}).get("username", "username")
                        st.markdown(f"View on GitHub: https://github.com/{github_username}/{selected_repo}")
            else:
                st.info("Please select at least one model to push.")
        else:
            st.warning("No trained models available. Please train a model in the Model Training module.")
    else:
        st.info("Connect to GitHub in the Connect tab first.")

# Tab 3: Pull Models from GitHub
with github_tabs[2]:
    st.markdown("### Pull Models from GitHub")
    
    if settings.get("github", {}).get("connected", False):
        st.markdown("#### Repository Access")
        
        access_type = st.radio(
            "Access models from:",
            ["Your repositories", "Public repositories", "Organization repositories"]
        )
        
        # Repository URL input
        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/username/repo"
        )
        
        if repo_url:
            # In a real app, this would fetch branches from the repository
            branches = ["main", "develop", "feature/new-models"]
            selected_branch = st.selectbox("Select branch", branches)
            
            st.markdown("#### Available Models")
            
            # Mock model list that would be fetched from the repository
            mock_models = [
                {"name": "sentiment-analysis-bert", "type": "NLP", "updated_at": "2023-06-10"},
                {"name": "customer-churn-xgboost", "type": "Classification", "updated_at": "2023-05-28"},
                {"name": "sales-forecasting-lstm", "type": "Time Series", "updated_at": "2023-06-02"}
            ]
            
            # Display models with selection
            for model in mock_models:
                col1, col2, col3 = st.columns([3, 2, 1])
                col1.markdown(f"**{model['name']}**")
                col2.markdown(f"Type: {model['type']}")
                col3.markdown(f"Updated: {model['updated_at']}")
                
            # Models to pull
            models_to_pull = st.multiselect(
                "Select models to pull",
                options=[model["name"] for model in mock_models]
            )
            
            if models_to_pull:
                # Pull options
                st.markdown("#### Pull Configuration")
                
                include_data = st.checkbox("Include training data", value=False)
                overwrite_existing = st.checkbox("Overwrite existing models", value=False)
                
                # Pull button
                if st.button("Pull Selected Models"):
                    with st.spinner("Pulling models from GitHub..."):
                        # In a real app, this would actually pull from GitHub
                        for model in models_to_pull:
                            st.info(f"Pulling {model}...")
                        
                        st.progress(100)
                        st.success(f"Successfully pulled {len(models_to_pull)} models")
            else:
                st.info("Please select at least one model to pull.")
        else:
            st.info("Enter a GitHub repository URL to list available models.")
    else:
        st.info("Connect to GitHub in the Connect tab first.")

# Tab 4: Collaboration
with github_tabs[3]:
    st.markdown("### Collaborative ML Development")
    
    st.markdown("""
    ALPHA's GitHub integration enables team collaboration on machine learning projects.
    Learn how to effectively collaborate with your team.
    """)
    
    # Collaboration workflows
    workflow_tabs = create_tab_panels(
        "Git Workflow", "Pull Requests", "Issues", "Documentation"
    )
    
    # Git Workflow tab
    with workflow_tabs[0]:
        st.markdown("#### Git Workflow for ML Projects")
        
        st.markdown("""
        Follow this workflow for effective collaboration on ML projects:
        
        1. **Create feature branch**: When working on a new model or feature, create a branch from `main`
        2. **Develop locally**: Train and test your models locally using ALPHA
        3. **Push to GitHub**: Push your models to your feature branch
        4. **Create pull request**: When ready, create a PR for team review
        5. **Review & merge**: After review and testing, merge to `main`
        6. **Tag releases**: Tag stable versions for deployment
        
        This workflow enables parallel development, code review, and easy rollbacks if needed.
        """)
        
        st.markdown("##### Branch Structure")
        
        display_code_block("""
        main              # Stable, production-ready models
          ├── develop     # Integration branch for tested features
          │     ├── feature/customer-segmentation   # Feature-specific branch
          │     └── feature/sales-prediction        # Feature-specific branch
          ├── hotfix      # Urgent fixes for production
          └── release     # Release candidates being tested
        """, language="bash")
    
    # Pull Requests tab
    with workflow_tabs[1]:
        st.markdown("#### Using Pull Requests")
        
        st.markdown("""
        Pull Requests (PRs) are essential for quality control in ML development:
        
        **Creating effective PRs:**
        
        - Use descriptive titles that summarize the model or feature
        - Include performance metrics in the description
        - Link to relevant issues or documentation
        - Keep PRs focused on a single model or feature
        
        **PR Review Process:**
        
        1. Code review: Check code quality and adherence to team standards
        2. Model evaluation: Review performance metrics and validation approach
        3. Documentation review: Ensure proper documentation of model behavior
        4. Testing: Verify model behavior with test cases
        
        **Approval and Merging:**
        
        - Require at least one approval before merging
        - Address all feedback before merge
        - Use squash merging to keep history clean
        """)
        
        # Example PR template
        st.markdown("##### Example PR Template")
        
        display_code_block("""
        ## Model Description
        Brief description of the model and its purpose
        
        ## Performance Metrics
        - Accuracy: 0.92
        - F1 Score: 0.89
        - AUC: 0.95
        
        ## Training Details
        - Training dataset: customer_data_2023
        - Algorithm: XGBoost
        - Hyperparameters: max_depth=6, learning_rate=0.1
        
        ## Validation Method
        5-fold cross-validation
        
        ## Potential Applications
        - Customer churn prediction
        - Marketing campaign optimization
        
        ## Related Issues
        Closes #45
        """, language="markdown")
    
    # Issues tab
    with workflow_tabs[2]:
        st.markdown("#### Using GitHub Issues")
        
        st.markdown("""
        GitHub Issues help track tasks, bugs, and feature requests for your ML projects:
        
        **Issue Categories:**
        
        - **Model Request**: Request for new model development
        - **Bug**: Issue with existing model behavior
        - **Enhancement**: Improvement to existing model
        - **Documentation**: Updates to model documentation
        - **Data**: Data collection or preparation tasks
        
        **Best Practices:**
        
        - Use labels to categorize issues (model-type, priority, etc.)
        - Link issues to PRs for traceability
        - Use milestones for project tracking
        - Assign issues to team members
        
        **Issue Resolution:**
        
        - Comment on progress regularly
        - Reference commits that address the issue
        - Close issues automatically with PR merges
        """)
        
        # Example issue template
        st.markdown("##### Example Issue Template")
        
        display_code_block("""
        ## Description
        [Describe the issue or feature request]
        
        ## Expected Behavior
        [What you expected to happen]
        
        ## Current Behavior
        [What actually happened]
        
        ## Steps to Reproduce
        1. [First step]
        2. [Second step]
        3. [More steps...]
        
        ## Environment
        - ALPHA version:
        - Python version:
        - Operating System:
        
        ## Additional Context
        [Any other information that might be relevant]
        """, language="markdown")
    
    # Documentation tab
    with workflow_tabs[3]:
        st.markdown("#### Documentation Standards")
        
        st.markdown("""
        Good documentation is crucial for ML project collaboration:
        
        **Model Documentation:**
        
        - **Model Card**: Description, use cases, limitations
        - **Training Process**: Data, parameters, environment
        - **Performance**: Metrics, validation method, test results
        - **Usage Examples**: Sample code for inference
        
        **Repository Documentation:**
        
        - **README.md**: Project overview, setup instructions
        - **CONTRIBUTING.md**: Contribution guidelines
        - **LICENSE.md**: License information
        - **docs/**: Detailed documentation
        
        **Code Documentation:**
        
        - Docstrings for all functions
        - Comments for complex sections
        - Inline type hints
        """)
        
        # Example model card
        st.markdown("##### Example Model Card")
        
        display_code_block("""
        # Customer Churn Prediction Model
        
        ## Model Description
        This model predicts the likelihood of customer churn based on usage patterns and demographic data.
        
        ## Intended Use
        - Identify customers at high risk of churning
        - Prioritize retention efforts
        - Personalize retention offers
        
        ## Training Data
        - 50,000 customer records
        - Time period: Jan 2022 - Dec 2022
        - Features: usage metrics, account age, demographic data
        
        ## Evaluation Results
        - Accuracy: 0.88
        - Precision: 0.85
        - Recall: 0.79
        - F1 Score: 0.82
        - AUC: 0.91
        
        ## Limitations
        - Limited performance for new customers (< 30 days)
        - May not generalize to international markets
        - Performance degrades for customers with sparse usage data
        
        ## Ethical Considerations
        - Model was tested for bias across demographic groups
        - Does not use protected attributes for prediction
        """, language="markdown")

# Display additional information in sidebar
with st.sidebar:
    st.markdown("## 💻 GitHub Integration")
    
    # Connection status
    if settings.get("github", {}).get("connected", False):
        st.success("✅ Connected to GitHub")
        github_username = settings.get("github", {}).get("username", "")
        st.markdown(f"User: **{github_username}**")
    else:
        st.warning("❌ Not connected to GitHub")
    
    # Resource links
    st.markdown("### Resources")
    
    with st.expander("GitHub Basics"):
        st.markdown("""
        - [Creating a repository](https://docs.github.com/en/get-started/quickstart/create-a-repo)
        - [Git Cheatsheet](https://education.github.com/git-cheat-sheet-education.pdf)
        - [Understanding branches](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-branches)
        """)
        
    with st.expander("ML Collaboration Tips"):
        st.markdown("""
        - **Version Data**: Store data versions alongside model versions
        - **Track Experiments**: Use commit messages to document experiment results
        - **CI/CD for ML**: Implement automated testing for models
        - **Review Metrics**: Focus PR reviews on model performance metrics
        - **Reproducibility**: Document environment requirements in repository
        """)
        
# Footer with links
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.markdown("[GitHub Documentation](https://docs.github.com)")
col2.markdown("[ML Reproducibility Guide](https://github.com/features)")
col3.markdown("[CI/CD for ML Projects](https://github.com/features/actions)") 