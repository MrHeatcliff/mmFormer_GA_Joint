import kagglehub

# Download latest version
path = kagglehub.dataset_download("sanglequang/brats2018")

print("Path to dataset files:", path)