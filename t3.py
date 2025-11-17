import os

# Check both source and destination
source_path = "d1_1000/test/not_hot_dog"
dest_path = "/Users/jaibhasin/Desktop/hot dog dataset/test/nothotdog"

if os.path.exists(source_path):
    source_count = len([f for f in os.listdir(source_path) 
                       if os.path.isfile(os.path.join(source_path, f))])
    print(f"Source (d1_1000/test/hot_dog): {source_count} files")

if os.path.exists(dest_path):
    dest_count = len([f for f in os.listdir(dest_path) 
                     if os.path.isfile(os.path.join(dest_path, f))])
    print(f"Destination (test/hotdog): {dest_count} files")