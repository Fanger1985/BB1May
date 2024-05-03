#include <NimBLEDevice.h>
#include <MPU6050.h>
#include "Wire.h"

// Pin Definitions for motor and sensor connections
#define IN1_LEFT 19
#define IN2_LEFT 15
#define IN1_RIGHT 5
#define IN2_RIGHT 18
#define TRIG_PIN 16
#define ECHO_PIN 17
#define MPU6050_SDA 21
#define MPU6050_SCL 22
#define IR_LEFT 13
#define IR_RIGHT 4
#define PIR_SENSOR_PIN 12

MPU6050 mpu; // Create MPU6050 object
int16_t ax, ay, az; // Acceleration
int16_t gx, gy, gz; // Gyroscope

// UUIDs for BLE services and characteristics
#define SERVICE_UUID "91bad492-b950-4226-aa2b-4ede9fa42f59"
#define MOTOR_CONTROL_UUID "01ae8a10-1127-42aa-9b23-82fae4d3c034"
#define SENSOR_READ_UUID "2a39b333-5b41-4e72-a774-6ed41a2062d2"

BLEServer *pServer = nullptr;
BLECharacteristic *pMotorControlCharacteristic;
BLECharacteristic *pSensorReadCharacteristic;

void setup() {
    Serial.begin(115200);
    Serial.println("Initializing BB1 Robot System...");

    NimBLEDevice::init("BB1 Robot");
    Serial.println("BLE device initialized with name 'BB1 Robot'");

    pServer = NimBLEDevice::createServer();
    Serial.println("BLE server created.");

    BLEService *pService = pServer->createService(SERVICE_UUID);
    Serial.println("BLE service created with UUID: " + String(SERVICE_UUID));

    pMotorControlCharacteristic = pService->createCharacteristic(
                                      MOTOR_CONTROL_UUID,
                                      NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::WRITE
                                  );
    Serial.println("Motor control characteristic set up.");

    pSensorReadCharacteristic = pService->createCharacteristic(
                                      SENSOR_READ_UUID,
                                      NIMBLE_PROPERTY::NOTIFY
                                  );
    Serial.println("Sensor read characteristic set up.");

    pService->start();
    NimBLEDevice::getAdvertising()->start();
    Serial.println("BLE advertising started.");

    // Initialize pin modes for all motors and sensors
    pinMode(IN1_LEFT, OUTPUT);
    pinMode(IN2_LEFT, OUTPUT);
    pinMode(IN1_RIGHT, OUTPUT);
    pinMode(IN2_RIGHT, OUTPUT);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(IR_LEFT, INPUT);
    pinMode(IR_RIGHT, INPUT);
    pinMode(PIR_SENSOR_PIN, INPUT);

    Wire.begin(MPU6050_SDA, MPU6050_SCL);
    mpu.initialize();
    if (!mpu.testConnection()) {
        Serial.println("MPU6050 connection failed.");
    } else {
        Serial.println("MPU6050 connection successful.");
    }
}

void loop() {
    static bool oldDeviceConnected = false;
    if (NimBLEDevice::getServer()->getConnectedCount()) {
        Serial.println("Device connected, processing commands.");
        if (pMotorControlCharacteristic->getValue().length() > 0) {
            handleCommand(pMotorControlCharacteristic->getValue().c_str());
            pMotorControlCharacteristic->setValue("");
        }
        // Add autonomous behavior and sensor updates here if needed
    }

    if (!NimBLEDevice::getServer()->getConnectedCount() && oldDeviceConnected) {
        NimBLEDevice::startAdvertising();
        Serial.println("Client disconnected, restarting advertising.");
        oldDeviceConnected = false;
    }

    if (NimBLEDevice::getServer()->getConnectedCount() && !oldDeviceConnected) {
        Serial.println("New client connected.");
        oldDeviceConnected = true;
    }
}

void handleCommand(String command) {
    Serial.println("Received command: " + command);
    if (command.startsWith("move ")) {
        if (command.endsWith("forward")) moveForward();
        else if (command.endsWith("backward")) moveBackward();
        else if (command.endsWith("left")) spinLeft();
        else if (command.endsWith("right")) spinRight();
    } else if (command == "stop") {
        stopMotors();
    } else if (command == "measure distance") {
        int distance = getUltrasonicDistance();
        pSensorReadCharacteristic->setValue(String(distance).c_str());
        pSensorReadCharacteristic->notify();
        Serial.println("Distance measured and notified: " + String(distance) + " cm");
    }
}

void moveForward() {
    digitalWrite(IN1_LEFT, HIGH);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, HIGH);
    digitalWrite(IN2_RIGHT, LOW);
    Serial.println("Moving forward.");
}

void moveBackward() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, HIGH);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, HIGH);
    Serial.println("Moving backward.");
}

void spinLeft() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, HIGH);
    digitalWrite(IN1_RIGHT, HIGH);
    digitalWrite(IN2_RIGHT, LOW);
    Serial.println("Spinning left.");
}

void spinRight() {
    digitalWrite(IN1_LEFT, HIGH);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, HIGH);
    Serial.println("Spinning right.");
}

void stopMotors() {
    digitalWrite(IN1_LEFT, LOW);
    digitalWrite(IN2_LEFT, LOW);
    digitalWrite(IN1_RIGHT, LOW);
    digitalWrite(IN2_RIGHT, LOW);
    Serial.println("Motors stopped.");
}

int getUltrasonicDistance() {
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    long duration = pulseIn(ECHO_PIN, HIGH);
    int distance = duration * 0.034 / 2;
    Serial.println("Ultrasonic distance measured: " + String(distance) + " cm");
    return distance;
}
