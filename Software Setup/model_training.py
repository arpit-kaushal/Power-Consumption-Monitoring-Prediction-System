import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """Load and preprocess the data with proper date parsing"""
    df = pd.read_csv(filepath)
    
    # Combine date and time with correct format (DD/MM/YYYY)
    df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], dayfirst=True)
    df = df.set_index('DATETIME')
    df = df.drop(['DATE', 'TIME'], axis=1)
    
    # Handle missing values
    if df.isnull().sum().any():
        df = df.interpolate()
    
    return df

def create_sequences(data, n_steps):
    """Create input-output sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

def build_lstm_model(n_steps, n_features):
    """Build and compile the LSTM model with proper serialization"""
    model = Sequential([
        LSTM(128, activation='relu', input_shape=(n_steps, n_features), return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_features)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=MeanSquaredError(),
        metrics=['mae']
    )
    return model

def main():
    # Configuration
    n_steps = 24  # 24-hour sequences
    test_size = 0.2
    random_state = 42
    features = ['POWER', 'UNITS']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('monitor_data.csv')
    data = df[features].values
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Create sequences
    X, y = create_sequences(scaled_data, n_steps)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False)
    
    # Build and train model
    print("Building and training model...")
    model = build_lstm_model(n_steps, len(features))
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save model with proper serialization
    print("Saving model...")
    save_model(model, 'power_units_lstm.keras')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/training_history.png')
    plt.close()
    
    print("Model training completed successfully!")

if __name__ == '__main__':
    main()