import sys
try:
    import kagglehub
except ImportError:
    import subprocess
    print("kagglehub not installed, installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

# Download latest version
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("mehaksingal/illegible-medical-prescription-images-dataset")

print("Path to dataset files:", path)
