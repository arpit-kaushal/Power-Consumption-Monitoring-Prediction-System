#define BLYNK_TEMPLATE_ID "TMPL357C2xJkG"
#define BLYNK_TEMPLATE_NAME "IoT Smart Energy Meter"
#define BLYNK_PRINT Serial

#include "EmonLib.h"
#include <EEPROM.h>
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

LiquidCrystal_I2C lcd(0x27, 16, 2); // I2C LCD address: 0x27, 16x2 display

// Define your Telegram bot token and chat ID
const char* telegramBotToken = "YOUR TELEGRAM BOT TOKEN";
const char* telegramChatID = "YOUR TELEGRAM CHAT ID";

// Constants for calibration (adjust these based on actual measurements)
const float vCalibration = 41.5 * (220.0 / 230.0); // Adjusted voltage calibration
const float currCalibration = 1.80 * (1.0 / 0.95); // Adjusted current calibration

// Blynk and WiFi credentials
const char auth[] = "e1ZYAx7QuolABU9itc4zt2EM7LJE7-dV";
const char ssid[] = "YOUR WIFI-HOTSPOT NAME";
const char pass[] = "YOUR WIFI-HOTSPOT PASSWORD";

// ThingSpeak credentials
const char* thingspeakApiKey = "YOUR THINGSPEAK API KEY"; // Updated ThingSpeak API Key
const char* thingspeakServer = "api.thingspeak.com";

// Google Script ID
const String GOOGLE_SCRIPT_ID = "YOUR GOOGLE SCRIPT ID";

// EnergyMonitor instance
EnergyMonitor emon;

// Timer for regular updates
BlynkTimer timer;

// Variables for energy calculation
float kWh = 0.0;
float cost = 0.0;
const float ratePerkWh = 6.5; // Cost rate per kWh
unsigned long lastMillis = millis();

// EEPROM addresses for each variable
const int addrKWh = 12;
const int addrCost = 16;

// Display page variable
int displayPage = 0;

// Reset button pin
const int resetButtonPin = 4; // Change to your button pin

// Function prototypes
void sendEnergyDataToBlynk();
void readEnergyDataFromEEPROM();
void saveEnergyDataToEEPROM();
void updateLCD();
void changeDisplayPage();
void sendBillToTelegram();
void resetEEPROM();
void sendDataToThingSpeak();
void sendDataToGoogleSheets(float voltage, float current, float power, float energy);

void setup()
{
  Serial.begin(115200);

  // Initialize WiFi
  WiFi.begin(ssid, pass);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected to WiFi");

  // Initialize Blynk
  Blynk.begin(auth, ssid, pass);

  // Initialize the LCD
  lcd.init();
  lcd.backlight();
  lcd.print("Initializing...");

  // Initialize EEPROM
  EEPROM.begin(32);

  // Initialize the reset button pin
  pinMode(resetButtonPin, INPUT_PULLUP);

  // Read stored data from EEPROM
  readEnergyDataFromEEPROM();

  // Setup voltage and current inputs
  emon.voltage(35, vCalibration, 1.7); // Voltage: input pin, calibration, phase_shift
  emon.current(34, currCalibration);    // Current: input pin, calibration

  // Setup timers
  timer.setInterval(2000L, sendEnergyDataToBlynk);
  timer.setInterval(2000L, changeDisplayPage);
  timer.setInterval(6000L, sendBillToTelegram);
  timer.setInterval(2000L, sendDataToThingSpeak); // Send data to ThingSpeak every 2 seconds
  timer.setInterval(2000L, []() {
    sendDataToGoogleSheets(emon.Vrms, emon.Irms, emon.apparentPower, kWh);
  }); // Send data to Google Sheets every 60 seconds
}

void loop()
{
  Blynk.run();
  timer.run();

  // Check if the reset button is pressed
  if (digitalRead(resetButtonPin) == LOW) // Assuming button press connects to ground
  {
    delay(200); // Debounce delay
    resetEEPROM();
  }
}

void sendEnergyDataToBlynk()
{
  emon.calcVI(20, 2000); // Calculate all. No. of half wavelengths (crossings), time-out

  float Vrms = emon.Vrms;
  float Irms = emon.Irms;
  float apparentPower = emon.apparentPower;

  // Calculate energy consumed in kWh
  unsigned long currentMillis = millis();
  kWh += apparentPower * (currentMillis - lastMillis) / 3600000000.0;
  lastMillis = currentMillis;

  // Calculate the cost based on the rate per kWh
  cost = kWh * ratePerkWh;

  // Save the latest values to EEPROM
  saveEnergyDataToEEPROM();

  // Send data to Blynk
  Blynk.virtualWrite(V0, Vrms);
  Blynk.virtualWrite(V1, Irms);
  Blynk.virtualWrite(V2, apparentPower);
  Blynk.virtualWrite(V3, kWh);
  Blynk.virtualWrite(V4, cost);

  // Update the LCD with the new values
  updateLCD();
}

void readEnergyDataFromEEPROM()
{
  EEPROM.get(addrKWh, kWh);
  EEPROM.get(addrCost, cost);

  // Initialize to zero if values are invalid
  if (isnan(kWh))
  {
    kWh = 0.0;
    saveEnergyDataToEEPROM();
  }
  if (isnan(cost))
  {
    cost = 0.0;
    saveEnergyDataToEEPROM();
  }
}

void saveEnergyDataToEEPROM()
{
  EEPROM.put(addrKWh, kWh);
  EEPROM.put(addrCost, cost);
  EEPROM.commit();
}

void updateLCD()
{
  lcd.clear();
  if (displayPage == 0)
  {
    lcd.setCursor(0, 0);
    lcd.printf("V: %.2fV I: %.2fA", emon.Vrms, emon.Irms);
    
    lcd.setCursor(0, 1);
    lcd.printf("P: %.2fW E: %.2fkWh", emon.apparentPower, kWh);
  }
  else if (displayPage == 1)
  {
    lcd.setCursor(0, 0);
    lcd.printf("Energy: %.2fkWh", kWh);
    
    lcd.setCursor(0, 1);
    lcd.printf("Cost: %.2f", cost);
  }
}

void changeDisplayPage()
{
  displayPage = (displayPage + 1) % 2;
  updateLCD();
}

void sendBillToTelegram()
{
  String message = "Total Energy Consumed: " + String(kWh, 2) + " kWh\nTotal Cost: ₹" + String(cost, 2);

  HTTPClient http;
  http.begin("https://api.telegram.org/bot" + String(telegramBotToken) + "/sendMessage");
  http.addHeader("Content-Type", "application/json");

  DynamicJsonDocument jsonDoc(256);
  jsonDoc["chat_id"] = telegramChatID;
  jsonDoc["text"] = message;

  String jsonString;
  serializeJson(jsonDoc, jsonString);

  int httpCode = http.POST(jsonString);

  // Optional: Handle HTTP errors here
  if (httpCode > 0) {
    Serial.println("Telegram message sent successfully");
  } else {
    Serial.println("Failed to send Telegram message");
  }

  http.end();
}

void resetEEPROM()
{
  kWh = 0.0;
  cost = 0.0;
  saveEnergyDataToEEPROM();
}

void sendDataToThingSpeak()
{
  HTTPClient http;
  
  // Construct the URL for ThingSpeak
  String url = "https://api.thingspeak.com/update?api_key=";
  url += thingspeakApiKey;
  url += "&field1=" + String(emon.Vrms);
  url += "&field2=" + String(emon.Irms);
  url += "&field3=" + String(emon.apparentPower);
  url += "&field4=" + String(kWh);
  url += "&field5=" + String(cost);

  // Begin the HTTP request
  http.begin(url);
  
  // Send the GET request
  int httpCode = http.GET();

  // Optional: Handle HTTP errors here
  if (httpCode > 0) {
    Serial.println("ThingSpeak update successful");
  } else {
    Serial.println("ThingSpeak update failed");
  }

  http.end();
}

void sendDataToGoogleSheets(float voltage, float current, float power, float energy)
{
  HTTPClient http;
  
  // Construct the URL for Google Sheets
  String url = "https://script.google.com/macros/s/" + GOOGLE_SCRIPT_ID + "/exec?";
  url += "voltage=" + String(voltage);
  url += "&current=" + String(current);
  url += "&power=" + String(power);
  url += "&units=" + String(energy);

  // Print the URL for debugging
  Serial.println("Google Sheets URL: " + url);

  // Begin the HTTP request
  http.begin(url);
  
  // Send the GET request
  int httpCode = http.GET();

  // Check the HTTP response code
  if (httpCode > 0) {
    Serial.println("Google Sheets update successful");
    // Print the response payload for debugging
    String payload = http.getString();
    Serial.println("Response: " + payload);
  } else {
    Serial.println("Google Sheets update failed");
  }

  // End the HTTP request
  http.end();
}
