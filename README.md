# RL Agent 1

## Overview

A reinforcement learning agent that learns to manipulate objects in a PyBullet simulation environment using Proximal Policy Optimization (PPO).

### What It Does
- Trains a robot manipulator to grab and place objects at target positions
- Uses PyBullet for physics simulation
- Implements PPO algorithm with TensorFlow/Keras
- Provides real-time training visualization

## Quick Start

### Setup

1. **Run the setup script:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Start training:**
   ```bash
   cd src
   python train.py
   ```

### Manual Setup (Alternative)

If you prefer to set up manually:

1. **Create virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   Note: On macOS, you may need to install pybullet with:
   ```bash
   CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet
   ```

## Training

### Basic Usage

```bash
cd src
python train.py
```

### Training Options

**With visualization (default):**
```bash
python train.py --episodes 500
```

**Without GUI (faster training):**
```bash
python train.py --episodes 500 --no-gui
```

**Custom training settings:**
```bash
python train.py --episodes 1000 --save-interval 50 --batch-size 128
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--episodes N` | Number of training episodes | 500 |
| `--batch-size N` | Batch size for training | 64 |
| `--no-gui` | Disable PyBullet GUI (faster) | False |
| `--save-interval N` | Save model every N episodes | 50 |

### Training Output

During training, you'll see:

1. **PyBullet Window** (if GUI enabled): Real-time visualization of the robot and object
2. **Metrics Plot**: Four live-updating plots:
   - Episode Rewards (with moving average)
   - Distance to Target
   - Training Losses (Actor and Critic)
   - Episode Length
3. **Console Output**: Progress updates every 10 episodes with:
   - Average rewards and episode lengths
   - Distance to target metrics
   - Training losses

### Model Saving

Models are automatically saved:
- Every N episodes (based on `--save-interval`) to `models/ppo_episode_N_actor.h5` and `models/ppo_episode_N_critic.h5`
- Final model saved as `models/ppo_final_actor.h5` and `models/ppo_final_critic.h5` at the end of training

## Inference

After training, test your model:

### Basic Inference
```bash
cd src
python inference.py --model models/ppo_final
```
This will use a random target position.

### Specify Target Position
```bash
python inference.py --model models/ppo_final --target-x 0.4 --target-y 0.2 --target-z 0.5
```

### Inference Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model PATH` | Path to trained model | `models/ppo_final` |
| `--no-gui` | Disable PyBullet GUI | False |
| `--target-x FLOAT` | Target X position | Random |
| `--target-y FLOAT` | Target Y position | Random |
| `--target-z FLOAT` | Target Z position | Random |

## Project Structure

```
rl_agent_1/
├── src/
│   ├── env.py          # PyBullet environment
│   ├── ppo.py           # PPO agent implementation
│   ├── train.py         # Training script
│   └── inference.py     # Inference script
├── models/              # Saved model weights
├── requirements.txt     # Python dependencies
├── setup.sh             # Setup script
└── README.md            # This file
```

## Requirements

- Python 3.8+
- TensorFlow 2.15
- PyBullet
- Gymnasium
- NumPy
- Matplotlib

## Troubleshooting

### macOS zlib Issue
If you encounter zlib errors on macOS when installing pybullet:
```bash
CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet
```

### PyBullet GUI Not Showing
If the GUI window doesn't appear, try:
- Running with `--no-gui` flag for headless mode
- Checking your display settings
- Using a remote desktop solution if on a headless server
