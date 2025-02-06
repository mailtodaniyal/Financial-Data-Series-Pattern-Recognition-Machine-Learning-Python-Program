import numpy as np
import pandas as pd
import joblib

def load_test_data(file_path):
    df = pd.read_csv(file_path)
    return df

def normalize_test_data(X):
    scaler = joblib.load("scaler.pkl")  
    return scaler.transform(X)

def recognize_patterns(test_file):
    df = load_test_data(test_file)
    X_test = df.values
    
    X_test = normalize_test_data(X_test)
    
    model = joblib.load("pattern_recognition_model.pkl")  
    predictions = model.predict(X_test) 
    
    label_mapping = joblib.load("pattern_labels.pkl")  
    reverse_mapping = {v: k for k, v in label_mapping.items()}  
    
    results = [reverse_mapping.get(p, "No pattern found") for p in predictions]
    
    for i, result in enumerate(results):
        print(f"Data row {i+1}: {result}")

recognize_patterns("test_data.csv")  
