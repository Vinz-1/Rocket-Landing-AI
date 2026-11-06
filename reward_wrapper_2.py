
import gymnasium as gym

class TimePenaltyWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward -= 0.001

        return observation, reward, terminated, truncated, info
