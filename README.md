# RL Agent 1

## 0️⃣ What Are We Solving? (Brief, Precise)

### Problem Statement

Build an open-ended embodied agent that:
- Accepts natural language (voice or text)
- Understands its own capabilities
- Decides whether to:
  - **Act** (control a robot in simulation)
  - **Respond** (explain why it can/can't act)
  - **Plan** (break task into sub-goals)
- Learns continuous control via RL
- Runs entirely locally on CPU
- Is agentic, not hardcoded
- Can later:
  - Plug into LangGraph
  - Swap simulators
  - Go sim → real robot

**This is Embodied Agent + RL + LLM Planning, not "robot follows commands".**

## 1️⃣ High-Level System (Mental Model)

```
User (Voice/Text)
   ↓
FunctionGemma (Intent + Capability Reasoning)
   ↓
Agent State Machine (later LangGraph)
   ↓
Skill Selection / Parameterization
   ↓
RL Policy (PPO, TensorFlow)
   ↓
PyBullet Simulator
   ↓
State / Reward / Metrics
   ↺
```

**LLM reasons, RL acts, simulator enforces physics.**

## 2️⃣ Installation

### Prerequisites

**⚠️ Important:** PyBullet has compatibility issues with newer macOS versions due to a zlib macro conflict. We use **Python 3.11** with a workaround.

### Setup Steps

**Option 1: Automated Setup (Recommended)**
```bash
./setup.sh
```

**Option 2: Manual Setup**

1. **Create a virtual environment with Python 3.11:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

2. **Upgrade pip and install build tools:**
   ```bash
   pip install --upgrade pip setuptools wheel cmake
   ```

3. **Install pybullet with workaround (macOS zlib fix):**
   ```bash
   CFLAGS="-Dfdopen=fdopen" CPPFLAGS="-Dfdopen=fdopen" pip install pybullet
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Troubleshooting

**PyBullet zlib error (`fdopen` macro conflict):**
- This is a known issue with pybullet's bundled zlib conflicting with macOS SDK headers
- The workaround above (using `CFLAGS` and `CPPFLAGS`) resolves this by redefining the macro
- If you encounter this error, make sure to install pybullet separately with the flags before installing other dependencies

**TensorFlow compatibility:**
- TensorFlow 2.15 requires Python 3.9-3.11
- Python 3.12+ requires TensorFlow 2.16+

## 3️⃣ Define Components

| Component | Role | Function |
|-----------|------|----------|
| FunctionGemma | Intent parsing, capability reasoning | Parses user intent and reasons about agent capabilities |
| Agent Core | Decide: act vs respond vs plan | Core decision-making logic for action selection |
| Skill Interface | Abstract robot abilities | Provides abstraction layer for robot skills |
| RL Policy (PPO) | Continuous motor control | Implements PPO algorithm for continuous control |
| PyBullet Env | Physics + state + reward | Physics simulation environment with state and reward computation |
| Metrics Layer | Success, efficiency, stability | Tracks performance metrics (success, efficiency, stability) |
| (Later) LangGraph | Agent memory & tool routing | Future integration for agent memory and tool routing |

## 4️⃣ Capability-Aware Agent

The robot knows what it can do.

### Capability Schema

```python
CAPABILITIES = {
    "move_end_effector": True,
    "grasp": True,
    "stack_objects": False,
    "navigate": False
}
```

### Example Reasoning

**User:**

"Stack three cubes"

**Agent:**

"I can grasp and place, but stacking is unstable. I can attempt two cubes. Should I proceed?"

This is not hardcoded — it's LLM + capability introspection.

## 5️⃣ FunctionGemma: Intent → Skill

### Function Definition

```python
functions = [
  {
    "name": "select_skill",
    "description": "Select robot skill",
    "parameters": {
      "type": "object",
      "properties": {
        "skill": {"type": "string"},
        "confidence": {"type": "number"}
      }
    }
  }
]
```

### Prompt Template

```
You are a robot.
These are your capabilities: {CAPABILITIES}
User command: {COMMAND}

Decide:
- respond_only
- act
- ask_clarification
```

### Output Example

```json
{
  "name": "select_skill",
  "arguments": {
    "skill": "pick_and_place",
    "confidence": 0.82
  }
}
```

## 6️⃣ PyBullet Environment (Baseline)

### Minimal Environment

```python
class RobotEnv:
    def __init__(self):
        self.physics = p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8)

    def reset(self):
        # load robot + object
        return obs

    def step(self, action):
        # apply joint deltas
        # compute reward
        return obs, reward, done, info
```

### Observation Vector

```
[ joint_pos (7),
  joint_vel (7),
  ee_xyz (3),
  object_xyz (3),
  gripper_state (1) ]
```

### Action Vector

```json
{
  "name": "select_skill",
  "arguments": {
    "skill": "pick_and_place",
    "confidence": 0.82
  }
}
```

## 7️⃣ PPO in TensorFlow

### Policy Network

```python
def policy_net(obs_dim, act_dim):
    inp = tf.keras.Input(shape=(obs_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    mean = tf.keras.layers.Dense(act_dim)(x)
    log_std = tf.Variable(-0.5 * tf.ones(act_dim))
    return tf.keras.Model(inp, mean), log_std
```

### Value Network

```python
def value_net(obs_dim):
    inp = tf.keras.Input(shape=(obs_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inp)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    value = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inp, value)
```

## 8️⃣ Training Loop

```python
for episode in range(episodes):
    obs = env.reset()
    traj = []

    while not done:
        action = policy(obs)
        next_obs, reward, done, info = env.step(action)
        traj.append((obs, action, reward))
        obs = next_obs

    update_ppo(traj)
```

**Use:**
- Small batch sizes
- Fewer epochs
- No parallel envs (CPU)

## 9️⃣ Metrics

### Core Metrics

| Metric | Meaning |
|--------|---------|
| Success Rate | Task completion |
| Episode Length | Efficiency |
| Reward Mean | Learning signal |
| Collision Count | Safety |
| Action Smoothness | Sim-to-real readiness |

### Logging

```python
metrics = {
  "success": success,
  "reward": ep_reward,
  "steps": steps
}
```

### Plot

```python
plt.plot(reward_history)
plt.title("Training Reward")
```