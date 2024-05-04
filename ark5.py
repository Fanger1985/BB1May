import aiohttp
import cv2
import asyncio
import numpy as np
import random
import logging
import threading

logging.basicConfig(filename='bb1_learning.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Configuration
esp32_base_url = "http://192.168.1.100/"
num_distance_states = 10  # Adjust based on the range of your ultrasonic sensor
presence_states = 2  # On or off states for presence detection
states = range(num_distance_states * presence_states)
actions = ['forward', 'backward', 'left', 'right', 'stop']
Q = np.zeros((len(states), len(actions)))
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_decay = 0.995  # Decay rate per episode
epsilon_min = 0.1  # Minimum exploration rate

async def send_http_get(session, endpoint):
    try:
        async with session.get(esp32_base_url + endpoint) as response:
            if response.status == 200:
                contentType = response.headers.get('Content-Type')
                if 'application/json' in contentType:
                    print(f"GET request to {endpoint} succeeded with JSON")
                    return await response.json()
                elif 'text/plain' in contentType:
                    text = await response.text()
                    print(f"GET request to {endpoint} succeeded with text: {text}")
                    return {"status": "success", "message": text}
                else:
                    print(f"Unexpected content type: {contentType}")
            else:
                print(f"GET request to {endpoint} failed with status {response.status}")
            return None
    except aiohttp.ClientError as e:
        print(f"HTTP request to {endpoint} threw an error: {e}")
        return None

def track_object():
    # Start capturing video from the USB camera
    cap = cv2.VideoCapture(0)
    tracker = cv2.TrackerCSRT_create()  # Creating a CSRT tracker
    _, frame = cap.read()  # Read the first frame
    bbox = cv2.selectROI("Tracking", frame, False)  # Let user select ROI
    tracker.init(frame, bbox)  # Initialize tracker with first frame and bounding box

    while True:
        _, frame = cap.read()
        success, bbox = tracker.update(frame)  # Update tracker
        if success:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        cv2.imshow("Tracking", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc pressed
            break
    cap.release()
    cv2.destroyAllWindows()

def start_tracking():
    tracking_thread = threading.Thread(target=track_object)
    tracking_thread.start()

async def get_state_from_sensors(session):
    response = await send_http_get(session, "sensors")
    if response:
        distance = response.get('distance', 100)
        ir_left = response.get('ir_left', 0)
        ir_right = response.get('ir_right', 0)

        distance_state = min(int(distance / 10), num_distance_states - 1)
        ir_state = 1 if ir_left > 0 or ir_right > 0 else 0
        state = distance_state + (num_distance_states * ir_state)
        return state
    else:
        print("Failed to retrieve sensor data")
        return 0  # Default state if no data retrieved

async def choose_action(session, state):
    global epsilon
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        action = actions[np.argmax(Q[state])]
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return action

async def simulate_get_reward(session, action):
    if action == 'forward':
        return 1
    elif action == 'backward':
        return -1
    else:
        return 0

async def update_q_table(state, action, reward, new_state):
    old_value = Q[state, actions.index(action)]
    future_optimal_value = np.max(Q[new_state])
    Q[state, actions.index(action)] = (1 - alpha) * old_value + alpha * (reward + gamma * future_optimal_value)

async def main():
    async with aiohttp.ClientSession() as session:
        state = await get_state_from_sensors(session)
        start_tracking()  # Start the tracking thread
        for episode in range(1000):
            action = await choose_action(session, state)
            await send_http_get(session, action)
            await asyncio.sleep(1)
            new_state = await get_state_from_sensors(session)
            reward = await simulate_get_reward(session, action)
            await update_q_table(state, action, reward, new_state)
            state = new_state
            logging.info(f"Episode {episode}: State={state}, Action={action}, Reward={reward}, Epsilon={epsilon}")
            if episode % 100 == 0:
                np.save('q_table.npy', Q)
                logging.info("Saved Q-table checkpoint")

if __name__ == "__main__":
    asyncio.run(main())
