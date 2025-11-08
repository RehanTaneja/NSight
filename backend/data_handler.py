import pandas as pd
import os

# save the uploaded dataset and return the local file path
def save_uploaded_data(file, upload_dir="uploads/data"):
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    return file_path

# load and return a pandas dataframe from a local file path
def load_data(file_path: str):
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        data = pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or JSON.")
    return data
