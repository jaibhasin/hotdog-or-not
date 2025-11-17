import kagglehub

# Download latest version
path = kagglehub.dataset_download("dansbecker/hot-dog-not-hot-dog")

print("Path to dataset files:", path)