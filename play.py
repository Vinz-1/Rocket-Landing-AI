
import gymnasium as gym
from stable_baselines3 import PPO

# Create Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Load our best-performing model
model = PPO.load("rocket_lander_time_penalty_2")

# Reset environment
observation, info = env.reset()

episode = 1

while True:
    # Predict action
    action, _states = model.predict(
        observation,
        deterministic=True
    )

    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)

    # Restart after episode ends
    if terminated or truncated:
        print(f"Episode {episode} finished!")
        episode += 1
        observation, info = env.reset()
