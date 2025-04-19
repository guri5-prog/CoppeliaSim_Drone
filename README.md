# 🚁 Drone Reinforcement Learning with CoppeliaSim + Stable-Baselines3

This project implements **reinforcement learning (RL)** for a drone (quadcopter) in a simulated 3D environment using **CoppeliaSim** and **Stable-Baselines3**. The drone learns to reach a goal position while avoiding obstacles and maintaining flight stability.

![Training Overview](https://your-image-or-demo-link.com) <!-- Optional: Add a gif or image demo -->

---

## 📦 Features

- ✅ **CoppeliaSim Remote Control** via ZMQ API
- ✅ Custom Gym-compatible drone environment
- ✅ **PPO** algorithm from Stable-Baselines3
- ✅ Episode logging and automatic checkpointing
- ✅ Real-time training plots for metrics and reward
- ✅ Handles crashing, hovering, and stuck detection

---

## 🔧 Requirements

- Python 3.7–3.10  
- [CoppeliaSim](https://www.coppeliarobotics.com/)  
- [ZMQ Remote API for CoppeliaSim](https://github.com/CoppeliaRobotics/zmqRemoteApi)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 🏁 Getting Started

### 1. Setup CoppeliaSim

- Open CoppeliaSim and load a scene that contains:
  - A drone (named `/Drone`)
  - A target dummy (`/TargetDummy`)
  - Optional obstacle blocks (`/ConcretBlock[0-15]`)
- Ensure **ZMQ Remote API** is enabled in CoppeliaSim.

### 2. Train the Drone

```bash
python train.py
```

Training will:
- Start or resume learning from the last checkpoint
- Save models every 5000 steps to `checkpoints/`
- Log episode stats to `training_log.csv`

You can interrupt training anytime with `Ctrl+C`.

---

## 📈 Visualizing Progress

To visualize training stats:

```bash
python -c "from drone_env import plot_training_log; plot_training_log()"
```

You'll see:
- Episode rewards
- Distance to target
- Final altitude
- Hover drift and stability
- Stuck/crash events

---

## 🧠 How It Works

The drone agent receives:
- Relative position and velocity to the target
- Orientation and angular velocity

It learns to:
- Reach the target
- Avoid crashing
- Maintain stable flight
- Not get stuck

Rewards are shaped to encourage efficient, safe, and smooth flying.

---

## 📁 Project Structure

```
.
├── train.py              # Main RL training script
├── drone_env.py          # Custom environment (Gym + CoppeliaSim)
├── checkpoints/          # Auto-saved models
├── training_log.csv      # Episode stats
├── requirements.txt      # Python dependencies
└── README.md             # You're here!
```

---

## 🔮 Possible Extensions

- Add curriculum learning
- Apply domain randomization for sim2real
- Test with different RL algorithms (SAC, TD3)
- Export for use with real drones (e.g., PX4, MAVSDK)

---

## 📜 License

MIT License © [Your Name](https://github.com/yourusername)

---

## 🙌 Acknowledgments

- [CoppeliaSim by Coppelia Robotics](https://www.coppeliarobotics.com/)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)


