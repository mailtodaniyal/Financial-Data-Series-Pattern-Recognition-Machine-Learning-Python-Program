import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def prepare_data(df):
    X = df.iloc[:, :-1].values  
    y = df.iloc[:, -1].values  
    
    X, scaler = normalize_data(X)
    joblib.dump(scaler, "scaler.pkl") 
    
    label_mapping = {label: i for i, label in enumerate(np.unique(y))}
    joblib.dump(label_mapping, "pattern_labels.pkl")  
    
    y = np.array([label_mapping[label] for label in y])  
    return X, y

def train_model(train_file):
    df = load_data(train_file)
    X, y = prepare_data(df)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, "pattern_recognition_model.pkl")  
    print("Model training complete and saved.")

train_model("training_data.csv")  
