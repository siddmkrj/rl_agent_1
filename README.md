# RL Agent 1

## 0️⃣ What Are We Solving? (Brief, Precise)

### Problem Statement
Build a reinforcement learning agent that can understand natural language commands from an LLM and execute object manipulation tasks in a simulated environment.

## Step 1: RL Agent with LLM Integration

### Overview
- Build an RL agent using PyBullet simulation and TensorFlow
- Environment contains a small object and a simple robot
- Train the robot to grab the object and place it on the left or right side based on commands received from an LLM
- LLM integration: Gemma and FunctionGemma

### Components
1. **Simulation Environment (PyBullet)**
   - Simple robot manipulator
   - Small object to be manipulated
   - Left and right placement zones

2. **RL Agent (TensorFlow)**
   - Deep reinforcement learning model
   - Trained to perform object grasping and placement
   - Responds to directional commands (left/right)

3. **LLM Integration**
   - Gemma: Natural language understanding
   - FunctionGemma: Function calling for command interpretation
   - Converts natural language commands to actionable instructions for the RL agent

## How to Start Training

### Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) If using LLM features, ensure Ollama is installed and running:
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull required models:
   ollama pull gemma2:2b
   ollama pull functiongemma
   ```

### Training the Agent

Navigate to the project root and run:

```bash
cd src
python train.py
```

#### Training Options

**Basic training with visualization:**
```bash
python train.py --episodes 500
```

**Training without GUI (faster):**
```bash
python train.py --episodes 500 --no-gui
```

**Training with custom settings:**
```bash
python train.py --episodes 1000 --save-interval 50 --llm-probability 0.7
```

**Training without LLM (random positions only):**
```bash
python train.py --episodes 500 --no-llm
```

#### Command Line Arguments

- `--episodes N`: Number of training episodes (default: 500)
- `--batch-size N`: Batch size for training (default: 64)
- `--no-gui`: Disable PyBullet GUI for faster training
- `--save-interval N`: Save model every N episodes (default: 50)
- `--no-llm`: Disable LLM command generation, use random positions
- `--llm-probability FLOAT`: Probability of using LLM command per episode (0.0-1.0, default: 0.7)

### What You'll See

1. **PyBullet Window**: Real-time visualization of the robot and object during training (if GUI enabled)
2. **Metrics Plot**: Four live-updating plots showing:
   - Episode Rewards (with moving average)
   - Distance to Target
   - Training Losses (Actor and Critic)
   - Episode Length

3. **Console Output**: Training progress with:
   - Average rewards and episode lengths
   - Distance to target metrics
   - Training losses
   - LLM commands being used (if enabled)

### Model Saving

Models are automatically saved:
- Every N episodes (based on `--save-interval`)
- Final model saved as `models/ppo_final` at the end of training

### Running Tests

Test the LLM integration:
```bash
python tests/test_llm_integration.py
```
