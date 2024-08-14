# File: config/repo_dir.py

import os
from pathlib import Path

REPO_NAME = 'bias_llm_clinical_challenge'

def get_repo_dir():
    """
    Find and return the repository directory.
    
    This function starts from the current working directory and
    moves up the directory tree until it finds a directory
    matching REPO_NAME.
    
    Returns:
        str: Absolute path to the repository directory.
    
    Raises:
        ValueError: If the repository directory is not found.
    """
    current_dir = Path.cwd().resolve()
    while current_dir != current_dir.parent:
        if current_dir.name == REPO_NAME:
            return str(current_dir)
        current_dir = current_dir.parent
    
    raise ValueError(f"Repository directory '{REPO_NAME}' not found")

# You can add more repository-related configurations or functions here