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