#include <Servo.h>

#define TRIG_PIN 5   // D1 (GPIO 5)
#define ECHO_PIN 4   // D2 (GPIO 4) - Use voltage divider
#define SERVO_PIN 0  // D3 (GPIO 0) - Servo control

Servo myServo;  // Create Servo object

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  
  myServo.attach(SERVO_PIN);  // Attach servo
  myServo.write(0);  // Start at 0 degrees
}

void loop() {
  // Trigger ultrasonic sensor
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo signal
  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // Timeout at 30ms (~5m range)

  float distance;
  if (duration == 0) {
    Serial.println("No Echo Received! Check connections.");
    distance = -1;  // Indicate an error
  } else {
    distance = (duration * 0.0343) / 2;  // Convert to cm
    Serial.print("Distance: ");
    Serial.print(distance);
    Serial.println(" cm");
  }

  // Control MG90S Servo based on distance
  if (distance > 0 && distance < 10) {
    myServo.write(90);  // Move to 90° if object is close
    Serial.println("Servo at 90°");
  } else {
    myServo.write(0);  // Default position
    Serial.println("Servo at 0°");
  }

  delay(500);  // Small delay before next measurement
}