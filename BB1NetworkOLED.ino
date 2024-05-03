#include <ESP8266WiFi.h>
#include <uMQTTBroker.h>
#include <Wire.h>
#include <SSD1306Wire.h>

// Define your WiFi credentials
#define WIFI_SSID "BB1-Network"
#define WIFI_PASSWORD "BB1-Secret"

// OLED settings
#define OLED_ADDRESS 0x3C
#define OLED_SDA D6
#define OLED_SCL D5
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64

// Define the IP address and subnet mask for the Access Point
IPAddress apIP(192, 168, 1, 1);
IPAddress subnet(255, 255, 255, 0);

// Create an instance of the broker
uMQTTBroker myBroker;

// Create an instance of the OLED display
SSD1306Wire display(OLED_ADDRESS, OLED_SDA, OLED_SCL);

void setup() {
    Serial.begin(115200);
    delay(1000); // Wait for serial monitor to open
    Serial.println("\n");

    // Start WiFi in Access Point mode
    WiFi.mode(WIFI_AP); // Set to Access Point mode
    WiFi.softAPConfig(apIP, apIP, subnet); // Set the IP address and subnet mask
    WiFi.softAP(WIFI_SSID, WIFI_PASSWORD); // Start the Access Point

    // Print the IP address of the Access Point
    Serial.print("Access Point IP address: ");
    Serial.println(WiFi.softAPIP());

    // Initialize the OLED display
    display.init();
    display.flipScreenVertically(); // Adjust if your display is flipped

    // Start the broker
    myBroker.init();

    Serial.println("Broker setup complete");
}

void loop() {
    // Display a message on the OLED display
    display.clear();
    display.setFont(ArialMT_Plain_16);
    display.drawString(0, 0, "BB1 Hub");

    // Maybe add some static display or fun graphics here since the connected device count isn't used
    display.setFont(ArialMT_Plain_10);
    display.drawString(0, 20, "Ready to relay!");
    
    display.display();

    // Handle any needed background tasks
    // In this simple setup, the loop can remain empty.
}
