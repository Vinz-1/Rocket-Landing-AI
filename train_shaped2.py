
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from reward_wrapper_2 import TimePenaltyWrapper

# Create environment with the same reward shaping
env = gym.make("LunarLander-v3")
env = TimePenaltyWrapper(env)

# Save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./shaped2_checkpoints/",
    name_prefix="continued_weak_penalty"
)

# Load our existing 100K trained model
model = PPO.load(
    "rocket_lander_time_penalty_2",
    env=env,
    device="cpu"
)

# Continue training for another 200K steps
model.learn(
    total_timesteps=200000,
    callback=checkpoint_callback,
    reset_num_timesteps=False
)

# Save separately to preserve the original model
model.save("rocket_lander_time_penalty_2_extended")

env.close()

print("Continued training completed!")
