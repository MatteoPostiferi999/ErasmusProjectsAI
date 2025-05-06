import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v3", render_mode='ansi')  # Change render_mode to 'human' for graphical visualization
env.reset()


# Function to evaluate a policy
def evaluate_policy(env, policy, number_of_episodes, number_of_steps):
    total_rewards = np.empty(number_of_episodes)
    total_actions = np.empty(number_of_episodes)

    for episode in range(number_of_episodes):
        state = env.reset()[0]
        episode_rewards = 0
        episode_steps = 0

        for t in range(number_of_steps):
            action = policy[state]
            new_state, reward, done, truncated, info = env.step(action)
            episode_rewards += reward
            episode_steps += 1

            if done:
                break
            else:
                state = new_state
        total_rewards[episode] = episode_rewards
        total_actions[episode] = episode_steps
        print('Episode {}: Total Reward: {}'.format(episode, episode_rewards))
        env.close()

    return np.mean(total_rewards), np.mean(total_actions)


# Q-learning
Qtable = np.zeros((500, 6))
number_of_episodes =10000
number_of_steps = 200
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1
epsilon_decay = 0.99

for episode in range(number_of_episodes):
    state = env.reset()[0]
    step = 0
    done = False
    episode_rewards = 0
    episode_steps = 0

    for interact in range(number_of_steps):
        if np.random.uniform(0, 1) > epsilon:
            action = np.argmax(Qtable[state, :])
        else:
            action = np.random.randint(0, 5)

        next_state, reward, done, truncated, info = env.step(action)

        Qtable[(state), action] = Qtable[state, action] + learning_rate * (reward + discount_factor *
        np.max(Qtable[(next_state), :]) - Qtable[state, action])

        state = next_state
        if done:
            break

    # Update epsilon for the next episode
    epsilon *= epsilon_decay

# Obtain policy from Qtable
policy = np.argmax(Qtable, axis=1)

average_reward, average_actions = evaluate_policy(env, policy, 10, 100)
print(f'Value function average_reward {average_reward:.4f}, average_number_of_actions {average_actions:.4f}')
