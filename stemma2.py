import time
import board
import busio
from adafruit_apds9960.apds9960 import APDS9960

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize APDS9960
apds = APDS9960(i2c)
apds.enable_proximity = True
apds.enable_color = True

try:
    while True:
        # Read from APDS9960
        r, g, b, c = apds.color_data
        print(f"Red: {r} Green: {g} Blue: {b} Clear: {c}")
        time.sleep(1)

except KeyboardInterrupt:
    print("Program exited cleanly")
