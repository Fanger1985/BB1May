import bluetooth
import asyncio
import logging
import random

logging.basicConfig(level=logging.INFO)

# Correct Bluetooth address of the ESP32
esp32_bt_address = "F896:AF67B4FC"  # Replace with your actual ESP32's Bluetooth address

behavior_scores = {
    'forward': 0,
    'left': 0,
    'right': 0,
    'stop': 0
}

async def send_bt_command(command, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            sock.connect((esp32_bt_address, 1))  # RFCOMM port 1
            sock.send(command + "\n")
            response = sock.recv(1024).decode('utf-8')  # Adjust buffer size as needed
            sock.close()
            return response
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            attempt += 1
            if attempt == retries:
                return None
        await asyncio.sleep(1)  # Wait a second before retrying

async def control_robot(command):
    response = await send_bt_command(command)
    update_behavior_score(command, response)
    return response

def update_behavior_score(command, response):
    if 'success' in response:
        behavior_scores[command] += 1
    elif 'failure' in response:
        behavior_scores[command] -= 1

async def choose_best_action():
    # Avoid actions with negative scores if possible
    best_action = max(behavior_scores, key=behavior_scores.get)
    return best_action if behavior_scores[best_action] > 0 else 'stop'

async def get_sensor_data():
    response = await send_bt_command("measure distance")
    if response:
        try:
            distance = int(response.split(": ")[1].strip().split(" ")[0])
            return {'distance': distance}
        except Exception as e:
            logging.error(f"Error parsing sensor data: {str(e)}")
            return None
    return {}

async def react_to_obstacle(distance):
    if distance < 30:
        action = await choose_best_action()
        await control_robot(action)
        await asyncio.sleep(1)
        await control_robot("stop")

async def monitor_environment():
    while True:
        sensor_data = await get_sensor_data()
        if sensor_data:
            distance = sensor_data.get("distance", 100)
            await react_to_obstacle(distance)
        await asyncio.sleep(1)

async def dance_routine():
    dance_moves = ["forward", "left", "forward", "right"]
    for move in dance_moves:
        await control_robot(move)
        await asyncio.sleep(1)
    await control_robot("stop")

async def express_emotion(emotion):
    logging.info(f"Emotion: {emotion}")
    await control_robot(f"expressEmotion {emotion}")

async def idle_behavior():
    while True:
        action = random.choice(["forward", "left", "right", "stop"])
        await control_robot(action)
        await asyncio.sleep(random.randint(1, 3))
        await control_robot("stop")

async def main():
    logging.info("Starting enhanced robot behavior script with adaptive learning")
    asyncio.create_task(monitor_environment())
    # Example of other behaviors
    await asyncio.gather(
        idle_behavior(),
        dance_routine()
    )

if __name__ == "__main__":
    asyncio.run(main())
