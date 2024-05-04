import aiohttp
import asyncio
import numpy as np
import random

# Configuration
esp32_base_url = "http://192.168.1.100/"
num_distance_states = 10
presence_states = 2
states = range(num_distance_states * presence_states)
actions = ['forward', 'backward', 'left', 'right', 'stop']
Q = np.zeros((len(states), len(actions)))
alpha = 0.1
gamma = 0.6
epsilon = 0.1

async def send_http_get(session, endpoint):
    async with session.get(esp32_base_url + endpoint) as response:
        if response.status == 200:
            print(f"GET request to {endpoint} succeeded")
            return await response.json()
        else:
            print(f"GET request to {endpoint} failed with status {response.status}")
            return None

async def choose_action(session, state):
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        action = actions[np.argmax(Q[state])]
    await send_http_get(session, action)
    return action

async def update_q_table(state, action, reward, new_state):
    old_value = Q[state, actions.index(action)]
    future_optimal_value = np.max(Q[new_state])
    Q[state, actions.index(action)] = (1 - alpha) * old_value + alpha * (reward + gamma * future_optimal_value)

async def main():
    async with aiohttp.ClientSession() as session:
        state = await get_state_from_sensors(session)
        for episode in range(1000):
            action = await choose_action(session, state)
            await asyncio.sleep(1)  # Simulate delay for state change
            new_state = await get_state_from_sensors(session)
            reward = await get_reward(session)
            await update_q_table(state, action, reward, new_state)
            state = new_state
            print(f"Episode {episode}: State={state}, Action={action}, Reward={reward}")
            if episode % 100 == 0:
                np.save('q_table.npy', Q)

if __name__ == "__main__":
    asyncio.run(main())
