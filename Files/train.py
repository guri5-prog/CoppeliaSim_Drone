from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from Files.drone_env import CoppeliaDroneEnv, plot_training_log  # 🔁 import the plotting function

# === Settings ===
checkpoint_dir = "./checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# === Helper to find latest checkpoint ===
def get_latest_checkpoint():
    pattern = re.compile(r"ppo_drone_model_(\d+)_steps\.zip")
    files = os.listdir(checkpoint_dir)
    checkpoints = [(int(m.group(1)), m.group(0)) for m in map(pattern.match, files) if m]
    if not checkpoints:
        return None, 0
    latest = max(checkpoints, key=lambda x: x[0])
    return os.path.join(checkpoint_dir, latest[1]), latest[0]

# === Custom Callback to Save Model Every N Steps ===
class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, start_step=0, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.start_step = start_step
        self.last_saved_step = start_step

    def _on_step(self) -> bool:
        total_steps = self.start_step + self.num_timesteps
        if total_steps - self.last_saved_step >= self.save_freq:
            filename = f"ppo_drone_model_{total_steps}_steps.zip"
            full_path = os.path.join(self.save_path, filename)
            self.model.save(full_path)
            self.model.save(os.path.join(self.save_path, "ppo_drone_model_latest.zip"))
            self.last_saved_step = total_steps
            print(f"💾 Saved checkpoint at step {total_steps} -> {filename}")
        return True

# === Create the environment ===
env = make_vec_env(CoppeliaDroneEnv, n_envs=1)

# === Load from checkpoint if exists ===
model_path, previous_steps = get_latest_checkpoint()
if model_path:
    print(f"🔁 Resuming from {model_path}")
    model = PPO.load(model_path, env=env)
else:
    print("🆕 Starting fresh training")
    model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05)  # Increased entropy coefficient for exploration

# === Custom Reward Function Updates ===
def _compute_reward(self):
    reward = 0.0

    # Distance to goal (progress reward)
    goal_vec = np.array(self.goal_position) - np.array(self.drone_position)
    dist = np.linalg.norm(goal_vec)
    progress = self._prev_dist_to_goal - dist
    reward += progress * 20.0  # Increased reward for progress towards goal

    # Penalize crashing (large penalty)
    if self._check_collision():
        reward -= 100.0  # Increased penalty for collision

    # Penalize being stuck
    vel = np.linalg.norm(self.drone_velocity)
    if vel < 0.1:
        self._stuck_counter += 1
        if self._stuck_counter > 20:  # Penalize more aggressively for getting stuck
            reward -= 5.0  # Increased penalty for getting stuck
    else:
        self._stuck_counter = 0

    # Slight reward for stable hovering near goal
    if dist < 1.0 and vel < 0.2:
        reward += 5.0  # Increased reward for stability near the target

    self._prev_dist_to_goal = dist
    return reward

# === Enhanced Plotting Function ===
def plot_training_log():
    """
    Plot the training log to visualize reward progression over time.
    Can be expanded to include more metrics.
    """
    if hasattr(model, 'logger'):
        reward_data = model.logger.get_stats('reward')  # Track reward stats
        if reward_data:
            steps = reward_data['x']
            rewards = reward_data['y']
            plt.plot(steps, rewards, label="Reward")
            plt.title("Training Progress: Rewards over Time")
            plt.xlabel("Steps")
            plt.ylabel("Reward")
            plt.legend()
            plt.show()

# === Continuous Training ===
save_freq = 5000
chunk_timesteps = 10000
callback = SaveCheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, start_step=previous_steps)

print("🚀 Continuous training started. Press Ctrl+C to stop.\n")

try:
    while True:
        model.learn(
            total_timesteps=chunk_timesteps,
            callback=callback,
            reset_num_timesteps=False
        )
        previous_steps += chunk_timesteps
        print(f"✅ Finished training chunk. Total steps: {previous_steps}")

        # 🔍 Plot progress after each training chunk
        try:
            plot_training_log()
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

except KeyboardInterrupt:
    print("\n🛑 Training interrupted. Saving final checkpoint...")
    model.save(f"{checkpoint_dir}/ppo_drone_model_interrupt_{previous_steps}_steps.zip")
    print("✅ Final model saved. Exiting safely.")
