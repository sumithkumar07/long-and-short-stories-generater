import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create the project directory structure."""
    # Define the directory structure
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "src/data",
        "src/models",
        "src/utils",
        "src/config",
        "notebooks",
        "tests"
    ]
    
    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create __init__.py files in Python package directories
        if directory.startswith("src"):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()

    # Create placeholder files
    placeholder_files = [
        "src/data/download_datasets.py",
        "src/data/process_datasets.py",
        "src/models/generator.py",
        "src/models/train.py",
        "src/config/config.py",
        "tests/__init__.py",
        "tests/test_generator.py",
        "tests/test_data_processing.py"
    ]
    
    for file_path in placeholder_files:
        Path(file_path).touch()

if __name__ == "__main__":
    create_directory_structure()
    print("Project directory structure created successfully!") 