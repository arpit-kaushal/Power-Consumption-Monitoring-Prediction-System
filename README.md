# Power Consumption Monitoring & Prediction System

This project presents a smart solution for real-time monitoring and forecasting of power and energy usage using an ESP32-based IoT setup integrated with cloud logging, machine learning, and a web dashboard.

---

## 🔌 Hardware Setup Guide

This section covers the hardware components, connection schematics, firmware deployment on ESP32, and linking the system to Blynk, ThingSpeak, and Telegram Bot for real-time monitoring and alerting.

---

### 🧰 Required Components

| Component         | Purpose                                      |
| ----------------- | -------------------------------------------- |
| ESP32 Dev Board   | Main microcontroller (Wi-Fi enabled)         |
| ACS712 Sensor     | Current measurement                          |
| ZMPT101B Sensor   | Voltage measurement                          |
| 16x2 LCD + I2C    | Real-time display of voltage, current, power |
| Breadboard, Wires | Circuit connections                          |
| 5V Power Supply   | Powers ESP32 and peripherals                 |

---

### 📌 Pin Connections

| Component      | ESP32 Pin    | Notes                      |
| -------------- | ------------ | -------------------------- |
| ACS712 (OUT)   | GPIO 34 (A0) | Analog input for current   |
| ZMPT101B (OUT) | GPIO 35 (A1) | Analog input for voltage   |
| LCD (SDA, SCL) | GPIO 21, 22  | I2C communication          |
| GND            | GND          | Common ground              |
| VCC (all)      | 5V / 3.3V    | Use logic-level compatible |

---

### ⚙️ Step-by-Step Setup

#### 1. 🚀 Upload Code to ESP32

**File**: `sketch_mar7a_copy_20250325014610.ino`

**Steps:**

1. Open Arduino IDE.
2. Install board support: **ESP32 by Espressif** via Board Manager.
3. Select board: `Tools → Board → ESP32 Dev Module`
4. Install necessary libraries:

   * `WiFi.h`
   * `HTTPClient.h`
   * `LiquidCrystal_I2C`
   * `EmonLib`
5. Edit the code to insert:

   * Your Wi-Fi SSID and password
   * Your Google Apps Script URL (for Sheets logging)
   * Blynk Auth Token (see next section)
   * ThingSpeak API Key
   * Telegram Bot token and chat ID
6. Connect ESP32 via USB and click ✅ Upload

---

#### 2. 📱 Connect to Blynk (IoT Mobile App)

**Steps:**

1. Download **Blynk IoT** app from Play Store/App Store.
2. Create a new project.
3. Add widgets:

   * 1x LCD display
   * 1x Gauge for current
   * 1x Gauge for voltage
   * 1x Gauge for power
   * 1x Notification (for alerts)
4. Copy the **Auth Token** from project settings.
5. Paste it into your ESP32 sketch (`char auth[] = "your_token";`)
6. Upload code again and monitor values on the app.

---

#### 3. 🌐 Send Data to ThingSpeak

**Steps:**

1. Sign in at [https://thingspeak.com](https://thingspeak.com)
2. Create a new channel with fields:

   * Voltage
   * Current
   * Power
3. Copy your **Write API Key**
4. In your ESP32 code, update:

   ```cpp
   String apiKey = "YOUR_API_KEY";
   ```
5. ThingSpeak will now log and plot data automatically.

---

#### 4. 🤖 Telegram Bot Integration (Alerts)

**Steps:**

1. Open Telegram and search for **BotFather**
2. Run `/newbot` and follow prompts to get your bot token
3. Note the **Bot Token** and update in code:

   ```cpp
   String botToken = "YOUR_BOT_TOKEN";
   ```
4. To get your `chat_id`:

   * Search for `userinfobot` on Telegram
   * Send `/start`, it will return your chat ID
   * Add that `chat_id` to your ESP32 code
5. Your ESP32 can now send alerts when:

   * Power exceeds a threshold
   * Sudden surges/drops are detected

---

#### 5. 📊 Google Sheets Logging

**Steps:**

1. Open Google Sheets → Extensions → Apps Script
2. Paste code from `logSensorData.gs`
3. Deploy as web app:

   * `Deploy → New deployment`
   * Choose type: Web App
   * Execute as: Me
   * Who has access: Anyone
   * Click `Deploy` → Copy URL
4. Update this URL in your ESP32 code:

   ```cpp
   const char* scriptUrl = "YOUR_DEPLOYED_SCRIPT_URL";
   ```
5. The ESP32 now pushes live data to your Google Sheet!

---

## 💻 Software Architecture & Usage Guide

This section explains how real-time power consumption data is collected, processed, used to train a machine learning model (LSTM), and how it is then used for forecasting future power usage through an interactive web app interface.

---

### 🗂 Project Directory Structure

```
Software-Setup/
│
├── app.py                       # Flask web app to run prediction & UI
├── model_training.py            # LSTM model training script
│
├── monitor_data.csv             # Historical training data (from Google Sheets)
├── current data.csv             # Real-time uploaded input for prediction
├── power_units_lstm.keras       # Trained LSTM model file
├── scaler.pkl                   # MinMaxScaler used to normalize input/output
│
├── static/                      # Auto-generated output files and assets
│   ├── training_history.png     # Loss curve of model training
│   ├── prediction_plot.png      # 48-hour forecast graph (PNG)
│   ├── uploaded_data.csv        # Saved original user-uploaded data
│   ├── prediction_data.csv      # Saved forecast data
│
├── templates/                   # Web UI HTML templates (Flask)
│   ├── index.html               # Main upload interface
│   ├── results.html             # Forecast display + AI Assistant chat
│
├── uploads/                     # Stores incoming user CSV files
└── README.md                    # Documentation (you are reading it)
```

---

### 🔁 Data Flow (End-to-End)

#### 🟩 1. Data Collection from Device

* **Source**: The ESP32 reads voltage and current values using ACS712 and ZMPT101B.
* **Processing**:

  ```text
  Power = Voltage × Current
  Units = Power × (duration / 1000)
  ```
* **Logging**: Data is sent to Google Sheets every few seconds using `logSensorData.gs`.
* Export the Sheet manually as `monitor_data.csv` for model training.

---

#### 🧠 2. Model Training with `model_training.py`

**Features Used**:

* POWER
* UNITS

**Preprocessing**:

* Combines `DATE` and `TIME` into a datetime index
* Interpolates missing values
* Normalizes values with `MinMaxScaler`
* Uses 24-hour sliding windows for time series

**Model**:

* LSTM (3-layer)
* Dropout regularization
* Optimizer: Adam
* Loss: Mean Squared Error
* Early stopping enabled

**Outputs**:

* `power_units_lstm.keras`: Saved model
* `scaler.pkl`: Scaler object
* `training_history.png`: Loss curve

**Run with**:

```bash
python model_training.py
```

---

#### 🌐 3. Web App (Prediction Interface)

**Backend**: `app.py` (Flask)

**Workflow**:

1. User uploads a `.csv` (e.g. `current data.csv`)
2. System:

   * Validates columns:

     ```
     DATE, TIME, VOLTAGE, CURRENT, POWER, UNITS
     ```
   * Extracts last 24 hours
   * Scales data using `scaler.pkl`
   * Predicts next 48 hours of `POWER` and `UNITS`
3. Saves:

   * Predictions to `prediction_data.csv`
   * Graph to `prediction_plot.png`
   * User data to `uploaded_data.csv`
4. Displays output on `results.html` with AI assistant

---

#### 🧠 AI Assistant (Optional)

* Integrated via **Gemini 1.5** API
* Handles queries like:

  * "What’s the expected usage tomorrow?"
  * "When is peak load?"
  * "How can I reduce power?"

---

### 🖥️ How to Run the Web App

**Install Dependencies**:

```bash
pip install flask pandas numpy tensorflow scikit-learn joblib matplotlib google-generativeai
```

**Launch**:

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

### 📤 Input File Format (Required Columns)

| Column  | Type   | Description              |
| ------- | ------ | ------------------------ |
| DATE    | string | Format: DD/MM/YYYY       |
| TIME    | string | Format: HH\:MM\:SS       |
| VOLTAGE | float  | Measured voltage         |
| CURRENT | float  | Measured current         |
| POWER   | float  | Calculated power (V × I) |
| UNITS   | float  | Energy units (kWh)       |

* **Minimum rows**: 24 (hourly records)

---

### 📊 Outputs

* **Graph**: Forecast of POWER and UNITS for 48 hours
* **Table**: Timestamped values
* **CSV**: Saved prediction results
* **Assistant**: Interprets trends and gives suggestions


