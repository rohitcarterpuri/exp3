import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ANNRegressor:
    def __init__(self, input_dim, config):
        self.input_dim = input_dim
        self.config = config
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build the ANN architecture"""
        model = keras.Sequential([
            # Input layer
            layers.Dense(self.config.INPUT_LAYER_NEURONS, 
                        activation='relu', 
                        input_shape=(self.input_dim,)),
            layers.Dropout(0.2),
            
            # Hidden layer 1
            layers.Dense(self.config.HIDDEN_LAYER_1_NEURONS, 
                        activation='relu'),
            layers.Dropout(0.2),
            
            # Hidden layer 2
            layers.Dense(self.config.HIDDEN_LAYER_2_NEURONS, 
                        activation='relu'),
            layers.Dropout(0.1),
            
            # Output layer
            layers.Dense(self.config.OUTPUT_LAYER_NEURONS)
        ])
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        print("Model built successfully")
        print(model.summary())
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train the ANN model"""
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data"""
        # Predict
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }
        
        print("\n=== Model Evaluation Metrics ===")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        
        return metrics, y_pred
    
    def save_model(self, path):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a saved model"""
        self.model = keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
