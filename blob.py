import asyncio
import random
import logging
from bleak import BleakClient

logging.basicConfig(level=logging.INFO)

esp32_bt_address = "FC:B4:67:AF:96:FA"  # The BLE address of the ESP32
SERVICE_UUID = "91bad492-b950-4226-aa2b-4ede9fa42f59"
MOTOR_CONTROL_UUID = "01ae8a10-1127-42aa-9b23-82fae4d3c034"
SENSOR_READ_UUID = "2a39b333-5b41-4e72-a774-6ed41a2062d2"

behavior_scores = {
    'forward': 0,
    'left': 0,
    'right': 0,
    'stop': 0
}

async def send_bt_command(client, command):
    try:
        await client.write_gatt_char(MOTOR_CONTROL_UUID, command.encode())
        response = await client.read_gatt_char(MOTOR_CONTROL_UUID)
        return response.decode('utf-8')
    except Exception as e:
        logging.error(f"BLE command error: {str(e)}")
        return None

async def control_robot(client, command):
    response = await send_bt_command(client, command)
    if response:
        update_behavior_score(command, response)
    return response

def update_behavior_score(command, response):
    if 'success' in response:
        behavior_scores[command] += 1
    elif 'failure' in response:
        behavior_scores[command] -= 1

async def choose_best_action():
    best_action = max(behavior_scores, key=behavior_scores.get)
    return best_action if behavior_scores[best_action] > 0 else 'stop'

async def get_sensor_data(client):
    response = await send_bt_command(client, "measure distance")
    try:
        distance = int(response.split(": ")[1].strip().split(" ")[0])
        return {'distance': distance}
    except Exception as e:
        logging.error(f"Error parsing sensor data: {str(e)}")
        return {}

async def react_to_obstacle(client, distance):
    if distance < 30:
        action = await choose_best_action()
        await control_robot(client, action)
        await asyncio.sleep(1)
        await control_robot(client, "stop")

async def monitor_environment(client):
    while True:
        sensor_data = await get_sensor_data(client)
        if sensor_data:
            distance = sensor_data.get('distance', 100)
            await react_to_obstacle(client, distance)
        await asyncio.sleep(1)

async def dance_routine(client):
    dance_moves = ["forward", "left", "forward", "right"]
    for move in dance_moves:
        await control_robot(client, move)
        await asyncio.sleep(1)
    await control_robot(client, "stop")

async def main():
    async with BleakClient(esp32_bt_address) as client:
        await client.connect()
        logging.info("Connected to ESP32")
        # Run tasks concurrently
        await asyncio.gather(
            monitor_environment(client),
            dance_routine(client),
            idle_behavior(client)
        )

if __name__ == "__main__":
    asyncio.run(main())
