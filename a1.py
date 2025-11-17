import kagglehub
import os
from pathlib import Path

# Download to cache
path = kagglehub.dataset_download("thedatasith/hotdog-nothotdog")

# Create symbolic link in current directory
link_name = Path.cwd() / "hotdog-nothotdog"
if not link_name.exists():
    os.symlink(path, link_name)
    print(f"Created symlink: {link_name} -> {path}")