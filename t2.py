import shutil
from pathlib import Path

source_base = Path("d1_1000/train")
dest_base = Path("train")

mappings = {
    "hot_dog": "hotdog",
    "not_hot_dog": "nothotdog"
}

for old_name, new_name in mappings.items():
    source = source_base / old_name
    destination = dest_base / new_name
    
    if source.exists():
        destination.mkdir(parents=True, exist_ok=True)
        
        # Get existing file count to continue numbering
        existing_files = [f for f in destination.iterdir() if f.is_file()]
        start_number = len(existing_files) + 1
        
        copied = 0
        for idx, file in enumerate(source.iterdir(), start=start_number):
            if file.is_file():
                # Get file extension
                suffix = file.suffix
                new_filename = f"{idx}{suffix}"
                dest_file = destination / new_filename
                
                shutil.copy2(str(file), str(dest_file))
                copied += 1
        
        print(f"Copied {old_name} -> {new_name}: {copied} files (numbered from {start_number})")