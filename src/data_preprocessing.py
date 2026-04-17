import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib

class DataPreprocessor:
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load the housing dataset"""
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        return df
    
    def explore_data(self, df):
        """Basic data exploration"""
        print("\n=== Data Info ===")
        print(df.info())
        print("\n=== Missing Values ===")
        print(df.isnull().sum())
        print("\n=== Statistical Summary ===")
        print(df.describe())
        
    def handle_missing_values(self, df, strategy='median'):
        """Handle missing values in the dataset"""
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy=strategy)
        df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
        
        print(f"Missing values after imputation: {df.isnull().sum().sum()}")
        return df
    
    def create_features(self, df):
        """Feature engineering"""
        # Example: Create polynomial features if needed
        # You can customize this based on your dataset
        
        # If there are date columns, extract year/month
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            df[f'{col}_year'] = pd.to_datetime(df[col]).dt.year
            df[f'{col}_month'] = pd.to_datetime(df[col]).dt.month
        
        return df
    
    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        self.feature_columns = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape}")
        print(f"Testing set size: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler or MinMaxScaler"""
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Features scaled using {self.scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def save_scaler(self, path):
        """Save the scaler for future use"""
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path):
        """Load a saved scaler"""
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")
