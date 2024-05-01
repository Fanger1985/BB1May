import aiohttp
import asyncio
import logging
import math

# Setup logging
logging.basicConfig(level=logging.INFO)

# Robot's base URL
esp32_base_url = "http://192.168.1.2/"

# Environment map to track visited positions and observed obstacles
environment_map = {}
current_orientation = 0  # Track the robot's current orientation in degrees

# Constants for movement correction
DESIRED_STRAIGHT_ORIENTATION = 0  # Adjust based on initial setup
ORIENTATION_CORRECTION_THRESHOLD = 5  # Degrees off from desired orientation to trigger correction

# Async HTTP request function with improved error handling
async def send_http_get(endpoint):
    url = f"{esp32_base_url}{endpoint}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logging.error(f"Failed to GET {url} with status {response.status}")
                    return None
    except aiohttp.ClientError as e:
        logging.error(f"HTTP client error when accessing {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

# Command control function for the robot with added logging
async def control_robot(command):
    result = await send_http_get(command)
    logging.info(f"Command '{command}' sent, response: {result}")
    update_orientation_after_command(command)

# Update orientation based on the command executed
def update_orientation_after_command(command):
    global current_orientation
    if command == "left":
        current_orientation = (current_orientation + 90) % 360
    elif command == "right":
        current_orientation = (current_orientation - 90) % 360

# Function to check and correct orientation if necessary
async def check_and_correct_orientation():
    if abs(current_orientation - DESIRED_STRAIGHT_ORIENTATION) > ORIENTATION_CORRECTION_THRESHOLD:
        if current_orientation > DESIRED_STRAIGHT_ORIENTATION:
            await control_robot("left")
        else:
            await control_robot("right")

# Fetch sensor and gyroscope data
async def get_robot_data():
    sensor_data = await send_http_get("sensors")
    gyro_data = await send_http_get("gyro")
    return sensor_data, gyro_data

# Obstacle reaction based on sensor data, taking into account all sensors
async def react_to_obstacle(sensor_data):
    front_distance = sensor_data.get("distance", 100)
    if front_distance < 30:
        await control_robot("stop")
        await asyncio.sleep(1)
        await control_robot("left" if math.sin(math.radians(current_orientation)) >= 0 else "right")
        await asyncio.sleep(1)
        await control_robot("stop")

# Monitor environment and respond to changes
async def monitor_environment():
    try:
        while True:
            sensor_data, gyro_data = await get_robot_data()
            update_orientation_from_gyro(gyro_data)
            current_position = gyro_data.get("position", {})
            environment_map[current_position] = sensor_data
            await react_to_obstacle(sensor_data)
            await check_and_correct_orientation()
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("Monitoring task was cancelled.")
        raise

# Update current orientation from gyro data
def update_orientation_from_gyro(gyro_data):
    global current_orientation
    # Assuming 'gz' is the angular rate around the Z-axis in degrees per second
    gz = gyro_data.get("gz", 0)
    current_orientation += gz * 0.05  # Assume update rate of 20 Hz (0.05 seconds per update)

# Main routine to handle async tasks
async def main():
    logging.info("Starting enhanced robot control script")
    monitor_task = asyncio.create_task(monitor_environment())
    try:
        await asyncio.sleep(3600)  # Keep the script running for an hour
    finally:
        monitor_task.cancel()
        await monitor_task

if __name__ == "__main__":
    asyncio.run(main())
