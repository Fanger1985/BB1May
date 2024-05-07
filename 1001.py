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
import board
import busio
from adafruit_apds9960.apds9960 import APDS9960
from adafruit_vl53l4cd import VL53L4CD

# Load pre-trained model and set up OpenCV for dog detection
model_path = 'frozen_inference_graph.pb'
config_path = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

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

# Setup I2C for sensors
i2c = busio.I2C(board.SCL, board.SDA)
apds = APDS9960(i2c)
apds.enable_proximity = True
apds.enable_color = True
tof = VL53L4CD(i2c)
tof.distance_mode = 2  # Set to Long Range Mode

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

def read_sensors():
    """Reads distance from TOF and color data from APDS9960."""
    distance = tof.distance
    r, g, b, c = apds.color_data
    logging.info(f"Distance: {distance} mm, Color - Red: {r}, Green: {g}, Blue: {b}, Clear: {c}")
    return distance, r, g, b, c

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

def start_face_tracking():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)  # Assuming your webcam index is 0

    if not cap.isOpened():
        logging.error("Cannot open the camera.")
        return  # Exit the function if camera can't be accessed

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture video frame")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

            for (x, y, w, h) in faces:
                logging.info(f"Face detected at x:{x}, y:{y}, w:{w}, h:{h}")
                # Optionally draw the rectangle around the face in the frame
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # If you need to show the frame, remove comments below (not recommended for headless)
            # cv2.imshow('Face Tracking', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    finally:
        cap.release()
        # cv2.destroyAllWindows()  # No need to call this on a headless system


async def get_state_from_sensors(session):
    distance, r, g, b, c = read_sensors()
    distance_state = min(int(distance / 10), num_distance_states - 1)
    # Here you can include logic to integrate color or other sensor data
    ir_state = 1 if (r + g + b + c) > 1000 else 0  # Example threshold
    return distance_state + (num_distance_states * ir_state)

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

# Indices for the objects of interest
husky_index = 252  # As identified for Siberian Husky
toilet_tissue_index = 1001  # Index for toilet tissue

# Your imports and setup are correct; make sure you define this function:
def recognize_object(frame, husky_index, toilet_tissue_index):
    input_tensor = np.expand_dims(frame, axis=0)  # Add batch dimension
    input_tensor = input_tensor.astype(np.float32)  # Ensure correct data type
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Extract output from batch
    
    recognized_husky = np.argmax(output_data) == husky_index
    recognized_toilet_tissue = np.argmax(output_data) == toilet_tissue_index

    if recognized_husky:
        logging.info("Siberian Husky detected!")
        return True
    elif recognized_toilet_tissue:
        logging.info("Toilet tissue detected!")
        return True

    return False


def dramatic_wiggle_servos():
    # A more dramatic sequence of movements for when an animal is spotted
    angles = [0, 90, 0, 90, 0]  # Quick movements to grab attention
    for angle in angles:
        set_servo_angle(left_servo, angle)
        set_servo_angle(right_servo, angle)
        time.sleep(0.3)  # Faster movement for more urgency

# Simplified robot_behavior function
async def robot_behavior(session):
    global epsilon
    load_q_table()
    object_found = False
    while True:
        state = await get_state_from_sensors(session)
        frame = camera_manager.capture_frame()
        if frame and recognize_object(frame, husky_index, toilet_tissue_index):
            if not object_found:  # Trigger dramatic actions only on first detection
                logging.info("Target object spotted! Initiating dramatic approach...")
                dramatic_wiggle_servos()  # Trigger the dramatic wiggle
                object_found = True  # Set to True so it won't wiggle again until reset
        else:
            object_found = False  # Reset the spotting flag
            logging.debug("No target object found, continuing exploration.")

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
        await robot_behavior(session)  # Only run robot_behavior if enhanced_exploration is not defined

if __name__ == "__main__":
    asyncio.run(main())

