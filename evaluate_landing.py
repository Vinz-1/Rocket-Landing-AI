
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import csv

MODEL = "rocket_lander_time_penalty_2.zip"
SEEDS = range(100, 150)

MAX_HORIZONTAL_SPEED = 0.15
MAX_VERTICAL_SPEED = 0.15
MAX_ANGLE = 0.15

env = gym.make("LunarLander-v3")
model = PPO.load(MODEL)

results = []

for seed in SEEDS:
    obs, info = env.reset(seed=seed)
    total_reward = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    x, y, vx, vy, angle, angular_velocity, left_leg, right_leg = obs

    stable = (
        left_leg == 1 and right_leg == 1
        and abs(vx) < MAX_HORIZONTAL_SPEED
        and abs(vy) < MAX_VERTICAL_SPEED
        and abs(angle) < MAX_ANGLE
    )

    if stable and terminated:
        outcome = "Stable Landing"
    elif terminated:
        outcome = "Crash/Unstable Termination"
    else:
        outcome = "Timeout"

    results.append([
        seed, round(total_reward, 2),
        outcome, round(x, 3), round(y, 3),
        round(vx, 3), round(vy, 3),
        round(angle, 3), round(angular_velocity, 3),
        int(left_leg), int(right_leg)
    ])

env.close()

with open("landing_diagnostics.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Seed", "Reward", "Outcome", "X", "Y",
        "VX", "VY", "Angle", "AngularVelocity",
        "LeftLeg", "RightLeg"
    ])
    writer.writerows(results)

print("\nDIAGNOSTICS COMPLETE")
print("Saved: landing_diagnostics.csv")

for outcome in ["Stable Landing", "Crash/Unstable Termination", "Timeout"]:
    count = sum(row[2] == outcome for row in results)
    print(f"{outcome}: {count}/{len(results)}")

print("\nSample failed episodes:")
for row in results:
    if row[2] != "Stable Landing":
        print(
            f"Seed {row[0]} | {row[2]} | "
            f"VX={row[5]} | VY={row[6]} | "
            f"Angle={row[7]} | Legs={row[9]},{row[10]}"
        )
