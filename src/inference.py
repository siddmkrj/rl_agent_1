import numpy as np
import argparse
import sys
import os

from env import RobotObjectEnv
from ppo import PPOAgent


def run_inference(model_path="models/ppo_final", gui=True, target_position=None):
    env = RobotObjectEnv(gui=gui)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first using train.py")
        env.close()
        return
    
    if target_position is None:
        workspace_bounds = env.get_workspace_bounds()
        x_min, x_max = workspace_bounds['x']
        y_min, y_max = workspace_bounds['y']
        z_min, z_max = workspace_bounds['z']
        target_position = [
            np.random.uniform(x_min, x_max),
            np.random.uniform(y_min, y_max),
            np.random.uniform(z_min, z_max)
        ]
        print(f"\nUsing random target position: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
    else:
        print(f"\nUsing specified target position: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
    
    state, info = env.reset(target_position=target_position)
    
    episode_reward = 0
    max_steps = 200
    
    print("\nExecuting command...")
    print("-" * 60)
    
    for step in range(max_steps):
        action = agent.select_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        
        state = next_state
        episode_reward += reward
        
        if terminated or truncated:
            break
        
        if gui:
            import time
            time.sleep(1.0 / 240.0)
    
    import pybullet as p
    object_pos_end, _ = p.getBasePositionAndOrientation(env.object_id)
    target_pos = info["target_position"]
    final_distance = np.linalg.norm(np.array(object_pos_end) - np.array(target_pos))
    
    print(f"\nEpisode completed!")
    print(f"Final distance to target: {final_distance:.3f}")
    print(f"Total reward: {episode_reward:.2f}")
    
    if final_distance < 0.1:
        print("✓ Success! Object placed at target location.")
    else:
        print("✗ Object not quite at target location.")
    
    if gui:
        print("\nPress Enter to close...")
        input()
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference with trained RL agent')
    parser.add_argument('--model', type=str, default="models/ppo_final", 
                       help='Path to trained model (default: models/ppo_final)')
    parser.add_argument('--no-gui', action='store_true', help='Disable PyBullet GUI')
    parser.add_argument('--target-x', type=float, default=None, 
                       help='Target X position')
    parser.add_argument('--target-y', type=float, default=None, 
                       help='Target Y position')
    parser.add_argument('--target-z', type=float, default=None, 
                       help='Target Z position')
    
    args = parser.parse_args()
    
    target_position = None
    if args.target_x is not None and args.target_y is not None and args.target_z is not None:
        target_position = [args.target_x, args.target_y, args.target_z]
    
    run_inference(
        model_path=args.model,
        gui=not args.no_gui,
        target_position=target_position
    )

