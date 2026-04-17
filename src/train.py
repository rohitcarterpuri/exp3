import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.model import ANNRegressor
import config

def plot_training_history(history, save_path='models/training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(history.history['mae'], label='Training MAE')
    axes[1].plot(history.history['val_mae'], label='Validation MAE')
    axes[1].set_title('Model MAE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training history plot saved to {save_path}")

def plot_predictions(y_test, y_pred, save_path='models/predictions_plot.png'):
    """Plot actual vs predicted values"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scatter plot
    axes[0].scatter(y_test, y_pred, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()], 
                 [y_test.min(), y_test.max()], 
                 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted Values')
    axes[0].grid(True)
    
    # Residual plot
    residuals = y_test - y_pred.flatten()
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Predictions plot saved to {save_path}")

def main():
    print("=== House Price Prediction using ANN ===\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    
    # Load data (you need to provide your dataset)
    df = preprocessor.load_data(config.DATA_PATH)
    
    # Explore data
    preprocessor.explore_data(df)
    
    # Handle missing values
    df = preprocessor.handle_missing_values(df)
    
    # Feature engineering
    df = preprocessor.create_features(df)
    
    # Assuming 'price' is the target column - modify based on your dataset
    target_column = 'price'  # Change this to your target column name
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_column, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Further split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=config.VALIDATION_SPLIT, 
        random_state=config.RANDOM_STATE
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    X_train_scaled, X_val_scaled = preprocessor.scale_features(X_train, X_val)
    
    # Save scaler
    preprocessor.save_scaler(config.SCALER_SAVE_PATH)
    
    print(f"\nTraining data shape: {X_train_scaled.shape}")
    print(f"Validation data shape: {X_val_scaled.shape}")
    print(f"Testing data shape: {X_test_scaled.shape}")
    
    # Step 2: Build and train model
    print("\nStep 2: Building and training ANN model...")
    ann_model = ANNRegressor(input_dim=X_train_scaled.shape[1], config=config)
    ann_model.build_model()
    
    # Train the model
    history = ann_model.train(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Step 3: Evaluate model
    print("\nStep 3: Evaluating model...")
    metrics, y_pred = ann_model.evaluate(X_test_scaled, y_test)
    
    # Step 4: Save model
    print("\nStep 4: Saving model...")
    ann_model.save_model(config.MODEL_SAVE_PATH)
    
    # Step 5: Plot results
    print("\nStep 5: Generating plots...")
    plot_training_history(history)
    plot_predictions(y_test, y_pred)
    
    print("\n=== Training Complete! ===")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"Scaler saved to: {config.SCALER_SAVE_PATH}")

if __name__ == "__main__":
    main()
