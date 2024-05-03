#include <MPU6050.h>
#include <BluetoothSerial.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLEService.h>
#include <BLECharacteristic.h>
#include <BLEAdvertisedDevice.h>

// Pin Definitions
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

// BLE UUIDs
#define SERVICE_UUID "91bad492-b950-4226-aa2b-4ede9fa42f59"
#define MOTOR_CONTROL_UUID "01ae8a10-1127-42aa-9b23-82fae4d3c034"
#define SENSOR_READ_UUID "2a39b333-5b41-4e72-a774-6ed41a2062d2"

BLEServer *pServer = nullptr;
BLECharacteristic *pMotorControlCharacteristic = nullptr;
BLECharacteristic *pSensorReadCharacteristic = nullptr;
BluetoothSerial ESP_BT;
MPU6050 mpu;

// Forward declarations for functions
void moveForward();
void moveBackward();
void spinLeft();
void spinRight();
void stopMotors();
int getUltrasonicDistance();
void handleCommand(String command);

// Callback class for handling BLE events
class MyCallbacks : public BLECharacteristicCallbacks {
void onWrite(BLECharacteristic *pCharacteristic) override {
    const char* value = pCharacteristic->getValue().c_str(); // Convert to const char*
    if (value) {
        Serial.println("Received over BLE:");
        Serial.println(value);
        handleCommand(String(value)); // Convert back to Arduino String if needed
    }
}

};

void setup() {
    Serial.begin(115200);
    BLEDevice::init("BB1 Robot");

    pServer = BLEDevice::createServer();
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pMotorControlCharacteristic = pService->createCharacteristic(
        MOTOR_CONTROL_UUID, 
        BLECharacteristic::PROPERTY_WRITE
    );
    pSensorReadCharacteristic = pService->createCharacteristic(
        SENSOR_READ_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pMotorControlCharacteristic->setCallbacks(new MyCallbacks());

    pService->start();
    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->start();

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
    // Handle BLE and autonomous behavior
    if (ESP_BT.available()) {
        String command = ESP_BT.readStringUntil('\n');
        handleCommand(command);
    }
    autonomousDecisionMaking();
    updateSensors();
}

void autonomousDecisionMaking() {
    int distance = getUltrasonicDistance();
    if (distance < 30) {
        stopMotors();
        ESP_BT.println("Stopped due to close obstacle.");
    }
}

void updateSensors() {
    // MPU6050 Reading
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
}

void handleCommand(String command) {
    // Handle motor and sensor commands
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
    int distance = duration * 0.034 / 2;
    return distance;
}
