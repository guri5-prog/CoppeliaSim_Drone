# === drone_env.py ===
import gym
import numpy as np
import time
import os
import csv
import matplotlib.pyplot as plt
import pandas as pd
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

LOG_FILE = "training_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "steps", "final_distance", "got_stuck", "final_z", "max_stuck"])

class CoppeliaDroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')

        self.drone = self.sim.getObject('/Drone')
        self.target = self.sim.getObject('/TargetDummy')
        self.script_handle = self.sim.getScript(self.sim.scripttype_childscript, self.drone)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=0, high=12, shape=(4,), dtype=np.float32)

        self.obstacles = []
        for i in range(16):
            try:
                self.obstacles.append(self.sim.getObject(f"/ConcretBlock[{i}]"))
            except:
                pass

        self.target_altitude = 1.0
        self.pid_error_sum = 0
        self.last_error = 0

        self.episode_reward = 0
        self.episode_step = 0
        self.episode_count = 0
        self._last_velocity_pos = None

        self._stuck_counter = 0
        self._max_stuck = 0
        self._last_positions = []
        self._last_z = 0.0
        self._start_time = time.time()

        if self.sim.getSimulationState() != self.sim.simulation_stopped:
            self.sim.stopSimulation()
            time.sleep(0.5)
        self.sim.startSimulation()
        time.sleep(0.5)

    def _check_collision(self):
        for obs in self.obstacles:
            if self.sim.checkCollision(self.drone, obs):
                return True
        return False

    def _get_obs(self):
        pos = self.sim.getObjectPosition(self.drone, -1)
        orient = self.sim.getObjectOrientation(self.drone, -1)
        target_pos = self.sim.getObjectPosition(self.target, -1)
        dist = np.linalg.norm(np.array(pos) - np.array(target_pos))
        stuck_flag = float(self._stuck_counter > 100)
        z_change = pos[2] - self._last_z
        self._last_z = pos[2]
        return np.array(pos + orient + target_pos + [dist, stuck_flag, z_change], dtype=np.float32)

    def _pid_hover_thrust(self, dt):
        pos = self.sim.getObjectPosition(self.drone, -1)
        error = self.target_altitude - pos[2]
        self.pid_error_sum += error * dt
        d_error = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        Kp, Ki, Kd = 15.0, 1.0, 3.0
        output = Kp * error + Ki * self.pid_error_sum + Kd * d_error
        return max(0, output)

    def _log_episode(self):
        final_pos = self.sim.getObjectPosition(self.drone, -1)
        final_target = self.sim.getObjectPosition(self.target, -1)
        final_dist = np.linalg.norm(np.array(final_pos) - np.array(final_target))
        got_stuck = int(self._stuck_counter > 100)
        final_z = final_pos[2]

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.episode_count, self.episode_reward, self.episode_step, final_dist, got_stuck, final_z, self._max_stuck])

        print(f"[EPISODE {self.episode_count}] Reward: {self.episode_reward:.2f}, Final Dist: {final_dist:.2f}, Stuck: {got_stuck}")

    def reset(self):
        if self.episode_step > 0:
            self._log_episode()

        self.episode_reward = 0
        self.episode_step = 0
        self.episode_count += 1
        self._stuck_counter = 0
        self._max_stuck = 0
        self._last_positions = []
        self._last_z = 0.5
        self._start_time = time.time()

        # Initialize the last position for velocity calculation
        self._last_pos = np.array(self.sim.getObjectPosition(self.drone, -1))

        self._collided = False

        # Reset the target and drone
        new_target_xy = np.random.uniform(-2.0, 2.0, size=2)
        self.sim.setObjectPosition(self.target, -1, [*new_target_xy, 1.0])
        self.sim.setObjectPosition(self.drone, -1, [0.0, 0.0, 0.5])
        self.sim.setObjectOrientation(self.drone, -1, [0.0, 0.0, 0.0])
        self.sim.resetDynamicObject(self.drone)

        self.pid_error_sum = 0
        self.last_error = 0
        for _ in range(20):
            thrust = self._pid_hover_thrust(0.02)
            self.sim.callScriptFunction('setMotorThrusts', self.script_handle, [], [thrust] * 4, [], '')
            time.sleep(0.02)

        # Initialize previous distance to goal
        target_pos = self.sim.getObjectPosition(self.target, -1)
        drone_pos = self.sim.getObjectPosition(self.drone, -1)
        self._prev_dist_to_goal = np.linalg.norm(np.array(target_pos) - np.array(drone_pos))

        return self._get_obs()

    def step(self, action):
        action = np.clip(action, 0.0, 12.0)
        self.sim.callScriptFunction('setMotorThrusts', self.script_handle, [], list(action), [], '')
        time.sleep(0.02)

        # Get the current observation
        obs = self._get_obs()

        # Compute velocity (difference in position)
        current_pos = np.array(self.sim.getObjectPosition(self.drone, -1))
        velocity = (current_pos - self._last_pos) / 0.02  # Dividing by time step (0.02 seconds)
        self._last_pos = current_pos  # Update last position

        # Compute reward based on current velocity
        reward = self._compute_reward(velocity)

        done = self._check_done(obs)

        self.episode_reward += reward
        self.episode_step += 1

        current_pos = np.array(self.sim.getObjectPosition(self.drone, -1))
        if self._last_velocity_pos is not None:
            self.drone_velocity = (current_pos - self._last_velocity_pos) / 0.02  # dt = 0.02s
        else:
            self.drone_velocity = np.array([0.0, 0.0, 0.0])
        self._last_velocity_pos = current_pos

        # Stuck counter logic
        self._last_positions.append(current_pos)
        if len(self._last_positions) > 10:
            self._last_positions.pop(0)

        if len(self._last_positions) >= 10:
            movement = np.max(np.linalg.norm(np.diff(self._last_positions, axis=0), axis=1))
            if movement < 0.01:
                self._stuck_counter += 1
            else:
                self._stuck_counter = 0

        self._max_stuck = max(self._max_stuck, self._stuck_counter)
        if self._stuck_counter > 60:
            done = True

        return obs, reward, done, {}

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def _compute_reward(self, velocity):
        reward = 0.0

        # Get the target position and drone position
        target_pos = self.sim.getObjectPosition(self.target, -1)
        drone_pos = self.sim.getObjectPosition(self.drone, -1)

        # Calculate distance between drone and target
        goal_vec = np.array(target_pos) - np.array(drone_pos)
        dist = np.linalg.norm(goal_vec)

        # Reward based on progress towards the goal
        reward += (self._prev_dist_to_goal - dist) * 20.0  # Reward for progress

        # Penalize for collision
        if self._check_collision() and not self._collided:
            reward -= 100.0
            self._collided = True

        # Penalize if the drone is stuck (low velocity)
        vel_magnitude = np.linalg.norm(velocity)
        if vel_magnitude < 0.1:
            self._stuck_counter += 1
            if self._stuck_counter > 20:  # Penalize more for being stuck
                reward -= 5.0
        else:
            self._stuck_counter = 0

        reward += 10.0 * np.exp(-dist)

        # Reward for stability near the goal
        if dist < 1.0 and vel_magnitude < 0.2:
            reward += 50.0  # Reward for stability near the target

        # Update the previous distance to goal for next step
        self._prev_dist_to_goal = dist

        return reward

    def _check_done(self, obs):
        pos = obs[0:3]
        orient = obs[3:6]
        if (abs(orient[0]) > 1.5 or abs(orient[1]) > 1.5 or pos[2] < 0.1 or pos[2] > 10.0 or abs(pos[0]) > 5 or abs(pos[1]) > 5):
            return True
        if time.time() - self._start_time > 120:
            return True
        return False

    def close(self):
        self.sim.stopSimulation()


def plot_training_log(log_path="training_log.csv"):
    if not os.path.exists(log_path):
        print("Log file not found.")
        return

    df = pd.read_csv(log_path)
    if len(df) < 5:
        return

    plt.clf()
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 3, 1)
    plt.plot(df["episode"], df["total_reward"], alpha=0.4)
    plt.plot(df["episode"], df["total_reward"].rolling(10).mean())
    plt.title("Reward per Episode")
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(df["episode"], df["final_distance"])
    plt.title("Final Distance to Target")
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(df["episode"], df["final_z"])
    plt.title("Final Altitude (Z)")
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(df["episode"], df["max_stuck"])
    plt.title("Max Stuck Counter")
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(df["episode"], df["got_stuck"])
    plt.title("Idle Hover Flag")
    plt.grid(True)

    plt.tight_layout()
    plt.pause(0.1)
    plt.show(block=False)