import os
from pathlib import Path

# Print diagnostic information
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())
print("Excel file exists:", Path("../National_Sales_ItemWise Report 2024-25 (1).xlsx").exists())

try:
    import pandas as pd
    import numpy as np
    from mlxtend.frequent_patterns import apriori
    from fastapi import FastAPI
    print("All required packages installed correctly")
except ImportError as e:
    print(f"Import error: {e}")

# Try to create a simple FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    print("Starting test server...")
    uvicorn.run("test_app:app", host="0.0.0.0", port=8000)