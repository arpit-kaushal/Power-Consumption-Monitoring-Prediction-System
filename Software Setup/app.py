from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import google.generativeai as genai
import warnings
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini API
try:
    genai.configure(api_key="AIzaSyATsTQFDTVrWddXFnOLVWIQic_-jrxLdz4")
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    print(f"Gemini API configuration failed: {e}")
    model_gemini = None

# Load models safely
try:
    model = load_model('power_units_lstm.keras')
    scaler = joblib.load('scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_new_data(filepath):
    """Preprocess new data with same format as training data"""
    try:
        df = pd.read_csv(filepath)
        
        # Check required columns
        required_cols = ['DATE', 'TIME', 'VOLTAGE', 'CURRENT', 'POWER', 'UNITS']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("CSV file doesn't contain all required columns")
        
        # Combine date and time with correct format
        df['DATETIME'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'], dayfirst=True)
        df = df.set_index('DATETIME')
        df = df.drop(['DATE', 'TIME'], axis=1)
        
        # Handle missing values
        if df.isnull().sum().any():
            df = df.interpolate()
            
        return df
    
    except Exception as e:
        raise ValueError(f"Error processing file: {str(e)}")

def create_sequences(data, n_steps):
    """Create input sequences for prediction"""
    X = []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
    return np.array(X)

def generate_plot(pred_df):
    """Generate prediction plot from the prediction DataFrame"""
    plt.figure(figsize=(12, 6))
    
    # Plot predicted values
    plt.plot(pred_df['PREDICTED_POWER'], label='Predicted Power', color='blue')
    plt.plot(pred_df['PREDICTED_UNITS'], label='Predicted Units', color='green')
    
    plt.title('48-Hour Power and Units Prediction')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    
    # Format x-axis
    plt.xticks(rotation=45)
    
    plot_path = 'static/prediction_plot.png'
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('error.html', message="Model not loaded properly")
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return render_template('error.html', message="No file uploaded")
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('error.html', message="No selected file")
        
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Preprocess new data
            new_df = preprocess_new_data(filepath)
            
            # Prepare data for prediction
            n_steps = 24
            features = ['POWER', 'UNITS']
            
            if len(new_df) < n_steps:
                return render_template('error.html', message="Uploaded file doesn't contain enough data (minimum 24 records needed)")
            
            # Get last n_steps records for prediction
            input_data = new_df[features].values[-n_steps:]
            
            # Scale data
            scaled_data = scaler.transform(input_data)
            
            # Predict next 48 hours (2 days)
            n_future = 48
            predictions = []
            current_sequence = scaled_data.copy()
            
            for _ in range(n_future):
                X = current_sequence[-n_steps:].reshape(1, n_steps, len(features))
                pred = model.predict(X, verbose=0)
                predictions.append(pred[0])
                current_sequence = np.vstack([current_sequence, pred[0]])
            
            # Inverse transform predictions
            predictions = scaler.inverse_transform(np.array(predictions))
            
            # Generate timestamps for predictions
            last_timestamp = new_df.index[-1]
            timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(n_future)]
            
            # Create results DataFrame
            pred_df = pd.DataFrame({
                'DATETIME': timestamps,
                'PREDICTED_POWER': predictions[:, 0],
                'PREDICTED_UNITS': predictions[:, 1]
            })
            
            # Generate plot using only predicted values
            plot_path = generate_plot(pred_df)
            
            # Prepare table data for rendering
            table_data = list(zip(
                pred_df['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                pred_df['PREDICTED_POWER'],
                pred_df['PREDICTED_UNITS']
            ))
            
            # Save prediction data for assistant
            pred_df.to_csv('static/prediction_data.csv', index=False)
            new_df.to_csv('static/uploaded_data.csv', index=True)
            
            return render_template('results.html', 
                                plot=plot_path,
                                table_data=table_data)
        else:
            return render_template('error.html', message="Invalid file type. Only CSV files are allowed")
    
    except Exception as e:
        return render_template('error.html', message=f"Prediction failed: {str(e)}")

@app.route('/ask_assistant', methods=['POST'])
def ask_assistant():
    if not model_gemini:
        return jsonify({"response": "AI assistant is currently unavailable."})
    
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        # Load the data
        pred_df = pd.read_csv('static/prediction_data.csv')
        uploaded_df = pd.read_csv('static/uploaded_data.csv')
        
        prompt = f"""
        You are a Power Consumption Analysis Assistant. Answer the user's question concisely and directly based on the data.
        
        Historical data (last 24 hours):
        {uploaded_df.tail(24).to_string()}
        
        Prediction data (next 48 hours):
        {pred_df.to_string()}
        
        User question: "{query}"
        
        Answer directly and specifically about the power predictions. 
        If asked about trends, mention specific values and time periods.
        If asked for recommendations, give one or two practical suggestions.
        Keep answers short (1-2 sentences max) and to the point.
        """
        
        response = model_gemini.generate_content(prompt)
        
        # Clean up the response
        clean_response = response.text.strip()
        if clean_response.startswith('"') and clean_response.endswith('"'):
            clean_response = clean_response[1:-1]
        
        return jsonify({"response": clean_response})
    
    except Exception as e:
        return jsonify({"response": "Sorry, I couldn't process your request."})

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=False)