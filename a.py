import kagglehub
import os

# Download into the current folder
target_path = os.path.abspath(".")

path = kagglehub.dataset_download(
    "dansbecker/hot-dog-not-hot-dog",
    path=target_path
)

print("Dataset saved to:", path)
