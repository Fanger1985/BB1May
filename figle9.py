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

class CameraManager:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logging.error("Cannot open the camera.")
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

def capture_frame():
    """Captures a single frame from the webcam and returns it."""
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        logging.error("Cannot open the camera.")
        return None

    try:
        ret, frame = cap.read()  # Read one frame from the camera
        if not ret:
            logging.error("Failed to capture frame from camera.")
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))  # Assuming your model expects 224x224 input
        return frame_resized
    finally:
        cap.release()  # Make sure to release the camera

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
        distance = response.get('distance', 100)
        ir_left = response.get('ir_left', 0)
        ir_right = response.get('ir_right', 0)
        distance_state = min(int(distance / 10), num_distance_states - 1)
        ir_state = 1 if ir_left > 0 or ir_right > 0 else 0
        return distance_state + (num_distance_states * ir_state)
    return 0  # Default state if no data

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
    """Directs the robot to explore more aggressively towards open spaces."""
    while True:
        state = await get_state_from_sensors(session)
        if state < 5:  # Assuming state < 5 indicates more open space
            await send_http_get(session, 'forward')
        elif random.random() > 0.8:
            await send_http_get(session, random.choice(['left', 'right']))
        await asyncio.sleep(1)

def recognize_object(frame, target_index):
    """Recognize specific objects in a frame."""
    # Ensure input tensor reshaping includes all necessary dimensions
    input_tensor = np.expand_dims(frame, axis=0)  # Add batch dimension
    input_tensor = input_tensor.astype(np.float32)  # Ensure correct data type

    # Set the tensor in the interpreter
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Process output data
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Extract output from batch

    # Check if the object recognized is the target object
    recognized = np.argmax(output_data) == target_index  # Define target_index based on your model's output
    return recognized


def wiggle_arms():
    """Wiggle the servos to celebrate finding the bottle."""
    # Define the angles for wiggling
    angles = [30, 60, 90, 60, 30, 0, 30]
    
    for angle in angles:
        set_servo_angle(left_servo, angle)
        set_servo_angle(right_servo, angle)
        time.sleep(0.2)  # Short delay to make the wiggle noticeable but quick

    # Reset servos to neutral position
    set_servo_angle(left_servo, 0)
    set_servo_angle(right_servo, 0)

async def robot_behavior(session):
    """Main behavior loop for the robot, applying Q-learning for decision making."""
    global epsilon
    load_q_table()  # Load the Q-table if it exists
    object_found = False

    while True:
        # Fetch both the hall sensor data and the general sensor state
        left_hall, right_hall = await get_hall_sensor_data(session)
        state = await get_state_from_sensors(session)  # Get current state from sensors

        # Check if BB1 is stuck based on hall sensor readings
        if left_hall == 0 and right_hall == 0:  # Assuming 0 implies no movement
            logging.info("BB1 seems to be stuck. Attempting to dislodge...")
            await send_http_get(session, 'backward')  # Try moving backward a bit
            await asyncio.sleep(1)  # Give BB1 time to move
            continue  # Skip the rest of the loop and recheck conditions

        # Check if it's time to hunt for the bottle
        if not object_found:
            frame = capture_frame()  # This needs definition based on your camera setup
            if recognize_object(frame, target='bottle'):
                logging.info("Bottle spotted! Initiating approach...")
                object_found = True
                await navigate_to_object(session)  # Navigate and interact with the bottle
            else:
                logging.debug("Bottle not found, continuing exploration.")

        # Regular Q-learning navigation if no specific task is being performed
        if not object_found:
            action_index = np.argmax(Q[state]) if random.random() > epsilon else random.randint(0, len(actions) - 1)
            action = actions[action_index]
            await send_http_get(session, action)
            new_state = await get_state_from_sensors(session)
            reward = calculate_reward(state, new_state, action)
            update_q_table(state, action_index, reward, new_state)
            state = new_state

            # Decay epsilon to reduce exploration over time
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Occasionally save the Q-table to disk
            if random.random() < 0.05:
                save_q_table()
                logging.info("Q-table saved.")

        # Randomly wiggle servos for liveliness
        if random.random() < 0.1:
            wiggle_servos()

async def navigate_to_object(session):
    """Navigate towards the bottle and perform interaction."""
    logging.info("Approaching the bottle...")
    await send_http_get(session, 'forward')
    time.sleep(2)  # Adjust timing based on your specific speed and distance needs
    await send_http_get(session, 'stop')
    proximity_sensor = await send_http_get(session, 'check_proximity')  # Assume this endpoint exists
    if proximity_sensor and int(proximity_sensor) < 10:  # Check if close enough and stop if too close
        await send_http_get(session, 'stop')
        logging.info("Close to object, stopped to avoid collision.")
    wiggle_arms()
    logging.info("Celebration wiggle executed!")

async def main():
    async with aiohttp.ClientSession() as session:
        threading.Thread(target=start_face_tracking).start()
        await asyncio.gather(robot_behavior(session), enhanced_exploration(session))

if __name__ == "__main__":
    asyncio.run(main())

