import aiohttp
import asyncio
import logging
import math
import gym
from gym import spaces
import numpy as np
import random
import pigpio

# Setup logging
logging.basicConfig(level=logging.INFO)

# Robot's base URL
esp32_base_url = "http://192.168.1.2/"

# Environment map to track visited positions and observed obstacles
environment_map = {}
current_orientation = 0  # Track the robot's current orientation in degrees

class RobotEnv(gym.Env):
    """Custom Environment that follows gym interface, now with expressive antennas!"""
    metadata = {'render.modes': ['console']}

    def __init__(self, pi):
        super(RobotEnv, self).__init__()
        self.pi = pi
        self.servo1_pin = 12  # Servo pin for the first antenna
        self.servo2_pin = 13  # Servo pin for the second antenna
        self.action_space = spaces.Discrete(9)  # Actions for servo positions
        self.observation_space = spaces.Box(low=0, high=255, shape=(3,), dtype=np.uint8)
        self.state = [100, 100, 100]
        self.done = False
        self.servo_positions = {0: 500, 1: 1500, 2: 2500}  # Example positions for 0, 90, and 180 degrees

    def step(self, action):
        reward = 0
        if action in [0, 1, 2]:  # Actions for servo 1
            new_angle1 = self.servo_positions[action]
            self.pi.set_servo_pulsewidth(self.servo1_pin, new_angle1)
        elif action in [3, 4, 5]:  # Actions for servo 2
            new_angle2 = self.servo_positions[action - 3]
            self.pi.set_servo_pulsewidth(self.servo2_pin, new_angle2)
        elif action in [6, 7, 8]:  # Actions for both servos
            new_angle1 = self.servo_positions[action - 6]
            new_angle2 = self.servo_positions[action - 6]
            self.pi.set_servo_pulsewidth(self.servo1_pin, new_angle1)
            self.pi.set_servo_pulsewidth(self.servo2_pin, new_angle2)
        return np.array(self.state), reward, self.done, {}

    def reset(self):
        self.state = [100, 100, 100]
        self.done = False
        return np.array(self.state)
    
    def render(self, mode='console'):
        if mode == 'console':
            print(f"State: {self.state}")

# Define the Q-learning Agent
class QLearningAgent:
    def __init__(self, action_space):
        self.q_table = {}
        self.action_space = action_space
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space.n - 1)
        else:
            return np.argmax(self.q_table.get(tuple(state), np.zeros(self.action_space.n)))

    def update_q_table(self, state, action, reward, next_state):
        old_value = self.q_table.get(tuple(state), np.zeros(self.action_space.n))[action]
        next_max = np.max(self.q_table.get(tuple(next_state), np.zeros(self.action_space.n)))
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[tuple(state)] = self.q_table.get(tuple(state), np.zeros(self.action_space.n))
        self.q_table[tuple(state)][action] = new_value

# Async HTTP request function
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
    except Exception as e:
        logging.error(f"HTTP error with {url}: {str(e)}")
        return None


# Command control function for the robot
async def control_robot(command):
    result = await send_http_get(command)
    logging.info(f"Command '{command}' sent, response: {result}")

# Fetch sensor and gyroscope data
async def get_robot_data():
    sensor_data = await send_http_get("sensors")
    gyro_data = await send_http_get("gyro")
    return sensor_data, gyro_data

# Enhanced gyro data processing
def process_gyro_data(gyro_data):
    gx, gy, gz = gyro_data.get("gx", 0), gyro_data.get("gy", 0), gyro_data.get("gz", 0)
    global current_orientation
    delta_t = 0.1
    current_orientation += gz * delta_t
    current_orientation %= 360
    return {"x": gx, "y": gy, "z": current_orientation}

# Main routine to handle async tasks
async def monitor_environment(pi):
    """Monitors sensor data to adjust actions or stop the robot if needed."""
    while True:
        # This could be any sensor checking logic
        sensor_status = await get_robot_data()  # Placeholder for sensor data fetching
        if sensor_status['error']:
            logging.warning("Sensor error detected, stopping robot.")
            pi.set_servo_pulsewidth(12, 0)
            pi.set_servo_pulsewidth(13, 0)
            break
        await asyncio.sleep(10)  # Check every 10 seconds

async def main():
    pi = pigpio.pi()  # Initialize the pigpio library to control GPIO
    env = RobotEnv(pi)
    agent = QLearningAgent(env.action_space)

    # Simulate training in the environment
    for _ in range(10000):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            env.render()

    # Start monitoring the environment
    logging.info("Starting enhanced robot control script")
    monitor_task = asyncio.create_task(monitor_environment(pi))
    try:
        await asyncio.sleep(3600)  # Keep running for an hour
    finally:
        monitor_task.cancel()
        await monitor_task

    pi.stop()  # Properly stop the pigpio instance when done

if __name__ == "__main__":
    asyncio.run(main())
