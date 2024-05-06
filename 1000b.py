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
from logging.handlers import RotatingFileHandler

# Setup basic logging
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
goal_state = None  # Adjust this based on your specific goal
obstacle_states = []  # Populate with any obstacle states if needed

# TensorFlow Lite model setup
model_path = 'mobilenet_v2_1.0_224.tflite'
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Suppress GPIO warnings
GPIO.setwarnings(False)

# GPIO setup for servos
GPIO.setmode(GPIO.BCM)
servo_left_pin = 12  # Left ear servo
servo_right_pin = 13  # Right ear servo
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
    with open("Q_table.json", "w") as f:
        json.dump(Q.tolist(), f)

def load_q_table():
    global Q
    try:
        with open("Q_table.json", "r") as f:
            Q = np.array(json.load(f))
    except FileNotFoundError:
        logging.info("No previous Q-table found, starting fresh.")

def calculate_reward(state, new_state, action):
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

def update_goal_state(new_goal_state):
    global goal_state
    goal_state = new_goal_state

async def send_http_get(session, endpoint):
    try:
        async with session.get(f"{esp32_base_url}{endpoint}") as response:
            if response.status == 200:
                return await response.json() if 'json' in response.headers.get('Content-Type', '') else await response.text()
            else:
                logging.error(f"Failed to execute {endpoint}: {response.status}")
                return None
    except Exception as e:
        logging.error(f"Error during HTTP GET to {endpoint}: {str(e)}")
        return None

async def get_state_from_sensors(session):
    response = await send_http_get(session, "sensors")
    if response:
        distance = response.get('distance', 100)
        ir_left = response.get('ir_left', 0)
        ir_right = response.get('ir_right', 0)
        distance_state = min(int(distance / 10), num_distance_states - 1)
        ir_state = 1 if ir_left > 0 or ir_right > 0 else 0
        return distance_state + (num_distance_states * ir_state)
    return 0

def set_servo_angle(servo, angle):
    duty = angle / 18 + 2
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo.ChangeDutyCycle(0)

def wiggle_servos():
    angles = [30, 60, 90, 60, 30, 0, 30]
    for angle in angles:
        set_servo_angle(left_servo, angle)
        set_servo_angle(right_servo, angle)

def recognize_object(frame, target_index):
    input_tensor = np.expand_dims(frame, axis=0)  # Add batch dimension
    input_tensor = input_tensor.astype(np.float32)  # Ensure correct data type
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Extract output from batch
    recognized = np.argmax(output_data) == target_index  # Define target_index based on your model's output

    # Additional dog detection logic
    if recognized:
        return True
    else:
        # Add logic to detect dogs using OpenCV
        dog_detected = False
        # Preprocess the frame for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

        # Set the input to the pre-trained model
        net.setInput(blob)

        # Run forward pass to detect objects in the frame
        detections = net.forward()

        # Loop over the detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Check if confidence exceeds a certain threshold (adjust as needed)
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])

                # Check if the detected object is a dog
                if class_id == 16:  # 16 is the class index for "dog"
                    dog_detected = True
                    break

        return dog_detected


async def robot_behavior(session):
    global epsilon
    load_q_table()
    object_found = False
    while True:
        left_hall, right_hall = await get_hall_sensor_data(session)
        state = await get_state_from_sensors(session)
        if left_hall == 0 and right_hall == 0:
            logging.info("BB1 seems to be stuck. Attempting to dislodge...")
            await send_http_get(session, 'backward')
            await asyncio.sleep(1)
            continue
        if not object_found:
            frame = camera_manager.capture_frame()
            if frame and recognize_object(frame, target_index=1):  # Adjust target_index accordingly
                logging.info("Bottle spotted! Initiating approach...")
                object_found = True
                await navigate_to_object(session)
            else:
                logging.debug("Bottle not found, continuing exploration.")
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

async def main():
    async with aiohttp.ClientSession() as session:
        threading.Thread(target=start_face_tracking).start()
        await asyncio.gather(robot_behavior(session), enhanced_exploration(session))

if __name__ == "__main__":
    asyncio.run(main())
