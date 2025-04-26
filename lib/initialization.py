import os
import shutil

def initialize_active_directory():
    """
    Initialize the active/ directory by copying files from template/ and creating necessary subdirectories.
    Ensures active/ is ready for prototyping CUDA kernels.
    """
    # Define source and destination directories
    template_dir = "template"
    active_dir = "active"
    
    # Create active/ if it doesn't exist
    os.makedirs(active_dir, exist_ok=True)
    
    # Create active/kernels/ and active/logs/
    os.makedirs(os.path.join(active_dir, "kernels"), exist_ok=True)
    os.makedirs(os.path.join(active_dir, "logs"), exist_ok=True)
    
    # Files to copy from template/ to active/
    files_to_copy = ["bindings.cu", "main.py", "setup.py"]
    
    # Copy files
    for file_name in files_to_copy:
        src_path = os.path.join(template_dir, file_name)
        dst_path = os.path.join(active_dir, file_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Warning: {src_path} not found")
    
    print("Initialization of active/ directory complete")