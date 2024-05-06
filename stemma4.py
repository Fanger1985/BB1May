import time
import board
import busio
from adafruit_apds9960.apds9960 import APDS9960
from adafruit_vl53l4cd import VL53L4CD

i2c = busio.I2C(board.SCL, board.SDA)

# Initialize APDS9960
apds = APDS9960(i2c)
apds.enable_proximity = True
apds.enable_color = True

# Initialize VL53L4CD
tof = VL53L4CD(i2c)
tof.distance_mode = 2  # Set to Long Range Mode

try:
    while True:
        # Read from VL53L4CD
        distance = tof.distance
        print(f"Distance: {distance} mm")

        # Read from APDS9960
        r, g, b, c = apds.color_data
        print(f"Red: {r} Green: {g} Blue: {b} Clear: {c}")

        time.sleep(1)

except KeyboardInterrupt:
    print("Program exited cleanly")
