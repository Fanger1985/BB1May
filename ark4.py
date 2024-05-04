import aiohttp
import asyncio
import numpy as np
import random
import logging
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
                print(f"GET request to {endpoint} succeeded")
                return await response.json()
            else:
                print(f"GET request to {endpoint} failed with status {response.status}")
                return None
    except aiohttp.ClientError as e:
        print(f"HTTP request to {endpoint} threw an error: {e}")
        return None

async def get_state_from_sensors(session):
    """Fetch sensor data from ESP32 and compute the current state."""
    response = await send_http_get(session, "sensors")
    if response:
        distance = response.get('distance', 100)  # Default to safe distance
        ir_left = response.get('ir_left', 0)  # Default to no obstacle
        ir_right = response.get('ir_right', 0)  # Default to no obstacle

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
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Update epsilon with decay
    return action

async def simulate_get_reward(session, action):
    """Simulate a reward function based on the action taken."""
    # Reward logic: positive for forward, negative for backward, neutral otherwise
    if action == 'forward':
        return 1  # Reward moving forward
    elif action == 'backward':
        return -1  # Penalize moving backward
    else:
        return 0  # Neutral for other actions

async def update_q_table(state, action, reward, new_state):
    old_value = Q[state, actions.index(action)]
    future_optimal_value = np.max(Q[new_state])
    Q[state, actions.index(action)] = (1 - alpha) * old_value + alpha * (reward + gamma * future_optimal_value)

async def main():
    async with aiohttp.ClientSession() as session:
        state = await get_state_from_sensors(session)
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
