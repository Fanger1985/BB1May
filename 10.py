import aiohttp
import asyncio
import cv2
import numpy as np
import random
import logging
import threading
import json
import RPi.GPIO as GPIO
import time
import board
import busio
from tflite_runtime.interpreter import Interpreter

# Setup basic logging
from logging.handlers import RotatingFileHandler

# Setup logging to file and console with rotating logs
log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
logFile = 'bb1_learning.log'

file_handler = RotatingFileHandler(logFile, mode='a', maxBytes=5*1024*1024, backupCount=2, encoding=None, delay=0)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.DEBUG)  # Change to INFO if too verbose

logging.basicConfig(handlers=[file_handler, console_handler], level=logging.DEBUG)

# Configuration
esp32_base_url = "http://192.168.1.100/"
num_distance_states = 10
presence_states = 2
states = range(num_distance_states * presence_states)
actions = ['forward', 'backward', 'left', 'right', 'stop', 'return_home']
Q = np.zeros((len(states), len(actions)))
alpha = 0.1
gamma = 0.6
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
home_base_location = None

# TensorFlow Lite model setup
model_path = 'mobilenet_v2_1.0_224.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Suppress GPIO warnings
GPIO.setwarnings(False)

# GPIO setup for servos
servo_left_pin = 12  # Left ear servo
servo_right_pin = 13  # Right ear servo
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_left_pin, GPIO.OUT)
GPIO.setup(servo_right_pin, GPIO.OUT)
left_servo = GPIO.PWM(servo_left_pin, 50)  # 50 Hz for servo
right_servo = GPIO.PWM(servo_right_pin, 50)
left_servo.start(0)
right_servo.start(0)

# Import sensor libraries
from adafruit_apds9960.apds9960 import APDS9960
from adafruit_vl53l4cd import VL53L4CD

# Initialize I2C
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize VL53L4CD
tof_sensor = VL53L4CD(i2c)
tof_sensor.distance_mode = 2  # Set to Long Range Mode

# Initialize APDS9960
apds_sensor = APDS9960(i2c)
apds_sensor.enable_proximity = True
apds_sensor.enable_color = True

class CameraManager:
    def __init__(self, retries=3, delay=1):
        self.cap = None
        for i in range(retries):
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                break
            logging.error(f"Attempt {i + 1}: Cannot open the camera, retrying...")
            time.sleep(delay)
        if not self.cap or not self.cap.isOpened():
            logging.error("Failed to open the camera after several attempts.")
            self.cap = None

    def capture_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret:
            logging.error("Failed to capture frame from camera.")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        return frame_resized

    def release(self):
        if self.cap:
            self.cap.release()


camera_manager = CameraManager()

def save_q_table():
    """Saves the Q-table to a file."""
    with open("Q_table.json", "w") as f:
        json.dump(Q.tolist(), f)

def load_q_table():
    """Loads the Q-table from a file."""
    global Q
    try:
        with open("Q_table.json", "r") as f:
            Q = np.array(json.load(f))
    except FileNotFoundError:
        logging.info("No previous Q-table found, starting fresh.")

def calculate_reward(state, new_state, action):
    """Calculate rewards based on the context of actions and results."""
    if new_state == goal_state:
        return 10  # Reward reaching a goal
    elif new_state in obstacle_states:
        return -10  # Penalty for hitting an obstacle
    elif new_state != state:
        return 1  # Standard reward for successful movement
    return -1  # Penalty for no movement or redundant movements

def update_q_table(state, action_index, reward, new_state):
    logging.debug(f"Updating Q-table from state {state} using action {actions[action_index]} with reward {reward}")
    old_value = Q[state, action_index]
    future_optimal_value = np.max(Q[new_state])
    new_value = old_value + alpha * (reward + gamma * future_optimal_value - old_value)
    Q[state, action_index] = new_value
    logging.debug(f"Updated Q-value from {old_value} to {new_value}")

home_base_location = None
goal_state = None  # Adjust this based on your specific goal
obstacle_states = []  # Populate with any obstacle states if needed

# Example: setting a dynamic goal state based on a sensor value
def update_goal_state(new_goal_state):
    global goal_state
    goal_state = new_goal_state


async def get_hall_sensor_data(session):
    """Retrieves hall sensor data from ESP32."""
    endpoint = "hall_sensors"
    hall_data = await send_http_get(session, endpoint)
    if hall_data:
        left_hall = hall_data.get('left_hall', 0)
        right_hall = hall_data.get('right_hall', 0)
        return left_hall, right_hall
    return 0, 0  # Default to 0 if no data

def set_servo_angle(servo, angle):
    """Set servo to a specific angle."""
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def wiggle_servos():
    """Wiggles both servos to indicate activity or emotion using safe servo control."""
    angles = [30, 60, 90, 60, 30, 0, 30]
    for angle in angles:
        set_servo_angle(left_servo, angle)
        set_servo_angle(right_servo, angle)

async def send_http_get(session, endpoint):
    logging.debug(f"Sending GET request to {endpoint}")
    try:
        async with session.get(f"{esp32_base_url}{endpoint}") as response:
            if response.status == 200:
                data = await response.json() if 'json' in response.headers.get('Content-Type', '') else await response.text()
                logging.debug(f"Received response from {endpoint}: {data}")
                return data
            else:
                logging.error(f"Failed to execute {endpoint}: {response.status}")
                return None
    except Exception as e:
        logging.error(f"Error during HTTP GET to {endpoint}: {str(e)}")
        return None

async def get_state_from_sensors(session):
    """Retrieves sensor data from ESP32 and calculates the current state index."""
    response = await send_http_get(session, "sensors")
    if response:
        distance = response.get('distance', 100)  # Assuming 'distance' key holds ultrasonic data
        ir_left = response.get('ir_left', 0)
        ir_right = response.get('ir_right', 0)
        distance_state = min(int(distance / 10), num_distance_states - 1)
        ir_state = 1 if ir_left > 0 or ir_right > 0 else 0
        current_state_index = distance_state + (num_distance_states * ir_state)
        logging.debug(f"Current Sensor State - Distance: {distance}mm, IR Left: {ir_left}, IR Right: {ir_right}, State Index: {current_state_index}")
        return current_state_index
    return 0  # Default state if no data

async def get_ultrasonic_distance(session):
    """Fetches ultrasonic distance from the ESP32."""
    try:
        response = await send_http_get(session, "ultrasonic_distance")
        if response:
            return int(response.get('distance', 0))
    except Exception as e:
        logging.error(f"Failed to fetch ultrasonic distance: {str(e)}")
    return None

def get_distance():
    """Get distance from VL53L4CD."""
    try:
        return tof_sensor.distance
    except Exception as e:
        logging.error(f"Error getting distance: {str(e)}")
        return None

async def continually_update_tof():
    while True:
        distance = get_distance()
        if distance:
            logging.info(f"Current TOF Distance: {distance}mm")
        await asyncio.sleep(0.1)  # Update every 100ms
    
def check_proximity():
    """Check proximity data from APDS9960 and return it."""
    _, proximity = get_color_proximity()  # Assuming this returns (color_data, proximity)
    return proximity

def get_color_proximity():
    """Get color and proximity data from APDS9960."""
    try:
        r, g, b, c = apds_sensor.color_data
        proximity = apds_sensor.proximity()  # Make sure this is not being called as proximity()
        logging.debug(f"Color and Proximity - Red: {r}, Green: {g}, Blue: {b}, Clear: {c}, Proximity: {proximity}")
        return (r, g, b, c), proximity
    except Exception as e:
        logging.error(f"Error getting color or proximity: {str(e)}")
        return (None, None, None, None), None


def start_face_tracking():
    """Starts a separate thread for tracking faces using the webcam."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    def tracking():
        global home_base_location
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(faces) > 0 and home_base_location is None:
                    home_base_location = {"x": faces[0][0], "y": faces[0][1]}
                    logging.info(f"Home base set at: {home_base_location}")
        except Exception as e:
            logging.error(f"Face tracking error: {str(e)}")
        finally:
            cap.release()

    threading.Thread(target=tracking, daemon=True).start()

async def navigate_to_home_base(session):
    """Navigates the robot back to the home base location using learned Q-values."""
    if home_base_location:
        current_state = await get_state_from_sensors(session)
        while current_state != home_base_location:
            action = actions[np.argmax(Q[current_state])]
            await send_http_get(session, action)
            current_state = await get_state_from_sensors(session)
            logging.info(f"Navigating back to home base, current state: {current_state}")
        logging.info("Returned to home base.")

async def enhanced_exploration(session):
    """Directs the robot to explore more aggressively towards open spaces, avoiding close obstacles."""
    while True:
        tof_distance = get_distance()
        ultrasonic_distance = await get_ultrasonic_distance(session)

        # Log current sensor readings for debugging and learning
        logging.debug(f"Exploration Readings - TOF: {tof_distance}mm, Ultrasonic: {ultrasonic_distance}mm")

        # Decision making based on sensor data
        if tof_distance is not None and tof_distance < 200:
            await send_http_get(session, 'backward')
            logging.info("Close TOF obstacle detected, moving backward.")
        elif ultrasonic_distance is not None and ultrasonic_distance < 400:
            await send_http_get(session, 'backward')
            logging.info("Close ultrasonic obstacle detected, moving backward.")
        else:
            # If path is clear, decide to move forward or adjust path
            action = determine_next_action()
            await send_http_get(session, action)
            logging.info(f"Path clear, moving {action}.")

        await asyncio.sleep(1)  # Control the loop frequency

def determine_next_action():
    """Determine the next action based on the current state and learned Q-values."""
    current_state = get_current_state()  # Assume function to fetch the current state index
    action_index = np.argmax(Q[current_state])
    return actions[action_index]


# Assuming 'bottle' is the third class in your model output
TARGET_INDEX_FOR_DOG = 2  # Change this to the actual index of 'dog' in your model output

def recognize_object(frame):
    """Recognize specific objects in a frame."""
    input_tensor = np.expand_dims(frame, axis=0)  # Add batch dimension
    input_tensor = input_tensor.astype(np.float32)  # Ensure correct data type

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Extract output from batch

    recognized = np.argmax(output_data) == TARGET_INDEX_FOR_DOG
    return recognized

def celebrate_with_dog():
    """Celebrate when finding a dog."""
    # Define the angles for excited wiggling
    angles = [20, 40, 60, 80, 100, 80, 60, 40, 20, 0]
    
    for angle in angles:
        set_servo_angle(left_servo, angle)
        set_servo_angle(right_servo, angle)
        time.sleep(0.1)  # Shorter delay for a more excited wiggle

    # Reset servos to neutral position
    set_servo_angle(left_servo, 0)
    set_servo_angle(right_servo, 0)

# Initialize flags
object_found = False

async def robot_behavior(session):
    global epsilon, object_found  # Ensure 'object_found' is known globally
    load_q_table()

    while True:
        distance = get_distance()
        ultrasonic_distance = await get_ultrasonic_distance(session)

        if distance and distance < 200 or (ultrasonic_distance is not None and ultrasonic_distance < 200):
            await send_http_get(session, 'stop')
            logging.info(f"Stopped due to close obstacle at {distance}mm.")
            continue  # Remain in the loop but skip moving commands until clear

        # If the obstacle is no longer detected, re-evaluate the situation
        if distance >= 200 and (ultrasonic_distance is None or ultrasonic_distance >= 200):
            logging.info("Path clear, reassessing movement strategy.")
            left_hall, right_hall = await get_hall_sensor_data(session)
            state = await get_state_from_sensors(session)

            if not object_found:
                frame = camera_manager.capture_frame()
                if frame and recognize_object(frame, target_index=TARGET_INDEX_FOR_DOG):
                    logging.info("Dog spotted! Initiating approach...")
                    object_found = True
                    await navigate_to_object(session)
                else:
                    logging.debug("Dog not found, continuing exploration.")

            if not object_found:
                action_index = np.argmax(Q[state]) if random.random() > epsilon else random.randint(0, len(actions) - 1)
                action = actions[action_index]
                await send_http_get(session, action)
                new_state = await get_state_from_sensors(session)
                reward = calculate_reward(state, new_state, action)
                update_q_table(state, action_index, reward, new_state)
                state = new_state

                epsilon = max(epsilon_min, epsilon * epsilon_decay)

                if random.random() < 0.05:
                    save_q_table()
                    logging.info("Q-table saved.")

            if random.random() < 0.1:
                wiggle_servos()


async def navigate_to_object(session):
    """Navigate towards the dog and perform interaction."""
    logging.info("Approaching the dog...")
    await send_http_get(session, 'forward')
    time.sleep(2)  # Adjust timing based on your specific speed and distance needs
    await send_http_get(session, 'stop')
    
    proximity = check_proximity()
    if proximity and int(proximity) < 10:  # Check if close enough and stop if too close
        await send_http_get(session, 'stop')
        logging.info("Close to object, stopped to avoid collision.")
    
    celebrate_with_dog()
    logging.info("Celebration wiggle executed!")

async def main():
    async with aiohttp.ClientSession() as session:
        threading.Thread(target=start_face_tracking).start()
        await asyncio.gather(robot_behavior(session), enhanced_exploration(session))

if __name__ == "__main__":
    asyncio.run(main())




