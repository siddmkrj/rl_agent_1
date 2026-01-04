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