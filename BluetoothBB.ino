#include <BluetoothSerial.h>
#include <Wire.h>
#include <MPU6050.h>

// Pin Definitions
#define IN1_LEFT 19
#define IN2_LEFT 15
#define IN1_RIGHT 5
#define IN2_RIGHT 18
#define TRIG_PIN 16
#define ECHO_PIN 17
#define MPU6050_SDA 21
#define MPU6050_SCL 22

// Bluetooth Serial object
BluetoothSerial ESP_BT;

// MPU6050 object
MPU6050 mpu;

// Function Declarations before they are used
void moveForward();
void moveBackward();
void spinLeft();
void spinRight();
void stopMotors();
int getUltrasonicDistance();

void setup() {
    Serial.begin(9600);
    if (!ESP_BT.begin("ESP32_BT")) {
        Serial.println("An error occurred initializing Bluetooth");
    } else {
        Serial.println("Bluetooth initialized");
        
        // Print the device's Bluetooth address
        uint64_t btAddress = ESP.getEfuseMac();  // Get device MAC address from EFUSE
        char btAddrStr[18];  // Create a string array for the MAC address
        sprintf(btAddrStr, "%04X:%08X", (uint16_t)(btAddress>>32), (uint32_t)btAddress);
        Serial.print("Bluetooth Address: ");
        Serial.println(btAddrStr);  // Print the Bluetooth address to the Serial Monitor
    }

    // Motor pins
    pinMode(IN1_LEFT, OUTPUT);
    pinMode(IN2_LEFT, OUTPUT);
    pinMode(IN1_RIGHT, OUTPUT);
    pinMode(IN2_RIGHT, OUTPUT);

    // Ultrasonic Sensor Pins
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    // Initialize MPU6050
    Wire.begin(MPU6050_SDA, MPU6050_SCL);
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed.");
    } else {
        Serial.println("MPU6050 connection successful.");
    }
}

void loop() {
    if (ESP_BT.available()) {
        String command = ESP_BT.readStringUntil('\n');
        if (command.startsWith("move ")) {
            if (command.endsWith("forward")) moveForward();
            else if (command.endsWith("backward")) moveBackward();
            else if (command.endsWith("left")) spinLeft();
            else if (command.endsWith("right")) spinRight();
        } else if (command == "stop") {
            stopMotors();
        } else if (command == "measure distance") {
            int distance = getUltrasonicDistance();
            ESP_BT.println("Distance: " + String(distance) + " cm");
        }
    }
    updateSensors();
    autonomousDecisionMaking();
}

void updateSensors() {
    // MPU6050 Reading
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    // You can make decisions based on these readings, like detecting falls or collisions.
}

void autonomousDecisionMaking() {
    int distance = getUltrasonicDistance();
    if (distance < 30) {
        stopMotors();  // Stop if something is too close
        ESP_BT.println("Stopped due to close obstacle.");
    }
}

void moveForward() {
    digitalWrite(IN1_LEFT, HIGH);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, HIGH);
    digitalWrite(IN2_RIGHT, LOW);
    ESP_BT.println("Moving forward");
}

void moveBackward() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, HIGH);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, HIGH);
    ESP_BT.println("Moving backward");
}

void spinLeft() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, HIGH);
    digitalWrite(IN1_RIGHT, HIGH);
    digitalWrite(IN2_RIGHT, LOW);
    ESP_BT.println("Spinning left");
}

void spinRight() {
    digitalWrite(IN1_LEFT, HIGH);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, HIGH);
    ESP_BT.println("Spinning right");
}

void stopMotors() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, LOW);
    ESP_BT.println("Motors stopped");
}

int getUltrasonicDistance() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH);
    int distance = duration * 0.034 / 2;  // Convert time to distance
    return distance;
}
