import aiohttp
import asyncio
import logging
import math

# Setup logging
logging.basicConfig(level=logging.INFO)

# Robot's base URL
esp32_base_url = "http://192.168.4.1/"

# Environment map to track visited positions and observed obstacles
environment_map = {}
current_orientation = 0  # Track the robot's current orientation in degrees

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

# Fetch sensor and gyroscope data
async def get_robot_data():
    sensor_data = await send_http_get("sensors")
    gyro_data = await send_http_get("gyro")
    return sensor_data, gyro_data

# Obstacle reaction based on sensor data
async def react_to_obstacle(sensor_data):
    front_distance = sensor_data.get("distance", 100)
    if front_distance < 30:
        logging.info("Obstacle detected! Stopping.")
        await control_robot("stop")
        await asyncio.sleep(1)
        # Decision based on simplistic rule, modify as needed:
        await control_robot("left")
        await asyncio.sleep(1)
        await control_robot("stop")

# Monitor environment and respond to changes
async def monitor_environment():
    try:
        while True:
            sensor_data, gyro_data = await get_robot_data()
            current_position = process_gyro_data(gyro_data)  # Process and use gyro data to update position
            environment_map[current_position] = sensor_data
            await react_to_obstacle(sensor_data)
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("Monitoring task was cancelled.")
        raise

# Enhanced gyro data processing to calculate orientation
def process_gyro_data(gyro_data):
    gx, gy, gz = gyro_data.get("gx", 0), gyro_data.get("gy", 0), gyro_data.get("gz", 0)
    logging.info(f"Gyro Data - gx: {gx}, gy: {gy}, gz: {gz}")

    # Calculate orientation based on gyroscope data assuming a simple integration over a fixed time step
    global current_orientation
    delta_t = 0.1  # Assuming data is processed every 100 ms, adjust based on your actual update rate
    current_orientation += gz * delta_t  # Integrate gyro z-axis to update orientation
    current_orientation %= 360  # Keep orientation within 0-359 degrees

    return {"x": gx, "y": gy, "z": current_orientation}  # Return calculated position and orientation

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
