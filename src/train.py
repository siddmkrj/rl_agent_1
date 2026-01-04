import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import argparse
import os
import sys
import tensorflow as tf

from env import RobotObjectEnv
from ppo import PPOAgent


class TrainingMetrics:
    def __init__(self):
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.distances_to_target = deque(maxlen=100)
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 10))
        self.fig.suptitle('Training Metrics', fontsize=16)
        
        self.axes[0, 0].set_title('Episode Rewards')
        self.axes[0, 0].set_xlabel('Episode')
        self.axes[0, 0].set_ylabel('Reward')
        
        self.axes[0, 1].set_title('Distance to Target')
        self.axes[0, 1].set_xlabel('Episode')
        self.axes[0, 1].set_ylabel('Distance')
        
        self.axes[1, 0].set_title('Training Losses')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Loss')
        
        self.axes[1, 1].set_title('Episode Length')
        self.axes[1, 1].set_xlabel('Episode')
        self.axes[1, 1].set_ylabel('Steps')
        
        plt.tight_layout()
        plt.ion()
        plt.show()
    
    def update(self, episode_reward, episode_length, distance_to_target, 
               actor_loss, critic_loss):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.distances_to_target.append(distance_to_target)
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
        self.axes[0, 0].clear()
        self.axes[0, 0].set_title('Episode Rewards')
        if len(self.episode_rewards) > 1:
            rewards = list(self.episode_rewards)
            self.axes[0, 0].plot(rewards, 'b-', alpha=0.5, label='Reward')
            if len(rewards) >= 10:
                moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
                self.axes[0, 0].plot(range(9, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)
        
        self.axes[0, 1].clear()
        self.axes[0, 1].set_title('Distance to Target')
        if len(self.distances_to_target) > 0:
            distances = list(self.distances_to_target)
            self.axes[0, 1].plot(distances, 'g-', alpha=0.5)
            if len(distances) >= 10:
                moving_avg = np.convolve(distances, np.ones(10)/10, mode='valid')
                self.axes[0, 1].plot(range(9, len(distances)), moving_avg, 'r-', linewidth=2)
            self.axes[0, 1].grid(True)
        
        self.axes[1, 0].clear()
        self.axes[1, 0].set_title('Training Losses')
        if len(self.actor_losses) > 0:
            actor_losses = list(self.actor_losses)
            critic_losses = list(self.critic_losses)
            self.axes[1, 0].plot(actor_losses, 'b-', label='Actor', alpha=0.7)
            self.axes[1, 0].plot(critic_losses, 'r-', label='Critic', alpha=0.7)
            self.axes[1, 0].legend()
            self.axes[1, 0].grid(True)
        
        self.axes[1, 1].clear()
        self.axes[1, 1].set_title('Episode Length')
        if len(self.episode_lengths) > 0:
            lengths = list(self.episode_lengths)
            self.axes[1, 1].plot(lengths, 'm-', alpha=0.5)
            if len(lengths) >= 10:
                moving_avg = np.convolve(lengths, np.ones(10)/10, mode='valid')
                self.axes[1, 1].plot(range(9, len(lengths)), moving_avg, 'r-', linewidth=2)
            self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)


def collect_trajectory(env, agent, max_steps=200):
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    
    import pybullet as p
    
    workspace_bounds = env.get_workspace_bounds()
    
    x_min, x_max = workspace_bounds['x']
    y_min, y_max = workspace_bounds['y']
    z_min, z_max = workspace_bounds['z']
    target_position = [
        np.random.uniform(x_min, x_max),
        np.random.uniform(y_min, y_max),
        np.random.uniform(z_min, z_max)
    ]
    
    state, info = env.reset(target_position=target_position)
    episode_reward = 0
    episode_length = 0
    
    target_pos = info["target_position"]
    
    for step in range(max_steps):
        action = agent.select_action(state, training=True)
        value = agent.get_value(state)
        
        mean, std = agent.actor(np.expand_dims(state, axis=0))
        from ppo import NormalDistribution
        dist = NormalDistribution(mean, std)
        action_tensor = tf.expand_dims(tf.constant(action, dtype=tf.float32), axis=0)
        log_prob = tf.reduce_sum(dist.log_prob(action_tensor), axis=1).numpy()[0]
        
        next_state, reward, terminated, truncated, info = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(terminated or truncated)
        log_probs.append(log_prob)
        values.append(value)
        
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    object_pos_end, _ = p.getBasePositionAndOrientation(env.object_id)
    distance_to_target = np.linalg.norm(np.array(object_pos_end) - np.array(target_pos))
    
    next_value = agent.get_value(state) if not (terminated or truncated) else 0
    returns, advantages = agent.compute_returns(rewards, dones, np.array(values), next_value)
    
    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'log_probs': log_probs,
        'returns': returns,
        'advantages': advantages,
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'distance_to_target': distance_to_target
    }


def train(episodes=500, batch_size=64, gui=True, save_interval=50):
    env = RobotObjectEnv(gui=gui)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(state_dim, action_dim)
    
    metrics = TrainingMetrics()
    
    os.makedirs('models', exist_ok=True)
    
    print("Starting training...")
    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
    print(f"Using random target positions")
    print("-" * 60)
    
    for episode in range(episodes):
        trajectory = collect_trajectory(env, agent)
        
        states = trajectory['states']
        actions = trajectory['actions']
        old_log_probs = trajectory['log_probs']
        returns = trajectory['returns']
        advantages = trajectory['advantages']
        
        actor_loss, critic_loss = agent.train(
            states, actions, old_log_probs, returns, advantages
        )
        
        metrics.update(
            trajectory['episode_reward'],
            trajectory['episode_length'],
            trajectory['distance_to_target'],
            actor_loss,
            critic_loss
        )
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(list(metrics.episode_rewards)[-10:])
            avg_length = np.mean(list(metrics.episode_lengths)[-10:])
            avg_distance = np.mean(list(metrics.distances_to_target)[-10:])
            
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Avg Length (last 10): {avg_length:.1f}")
            print(f"  Avg Distance to Target: {avg_distance:.3f}")
            print(f"  Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            print("-" * 60)
        
        if (episode + 1) % save_interval == 0:
            model_path = f"models/ppo_episode_{episode + 1}"
            agent.save(model_path)
            print(f"Model saved to {model_path}")
    
    env.close()
    agent.save("models/ppo_final")
    print("Training completed! Final model saved to models/ppo_final")
    
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RL agent with PPO')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--no-gui', action='store_true', help='Disable PyBullet GUI')
    parser.add_argument('--save-interval', type=int, default=50, help='Save model every N episodes')
    
    args = parser.parse_args()
    
    train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        gui=not args.no_gui,
        save_interval=args.save_interval
    )

