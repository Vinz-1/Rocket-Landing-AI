
# Rocket Landing AI 🚀

An autonomous lunar lander agent developed using Reinforcement Learning and Proximal Policy Optimization (PPO).

## Overview
This project explores autonomous landing behavior using Gymnasium's LunarLander-v3 environment and Stable-Baselines3.

## Experiments
- Baseline PPO
- Reward shaping using time penalties
- Progress-based reward experiments
- Orientation-based reward experiments
- Evaluation across 50 fixed random seeds
- Stable landing analysis

## Best Experimental Result
- Model: Mild time-penalty PPO
- Average reward: +98.74
- Stable landings: 21/50
- Stable landing rate: 42%

Stable landing criteria: both landing legs touching the ground, low horizontal and vertical velocities, and low angular deviation.

## Technologies
Python, Gymnasium, Stable-Baselines3, PPO, NumPy, Pandas, Matplotlib

## Future Work
- Improve landing success rate
- Analyze failed trajectories
- Test environmental robustness
- Optimize reward shaping
