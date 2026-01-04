import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt


class CartPoleTrainer:
    def __init__(self, episodes=1000, target_update_frequency=10):
        self.env = gym.make('CartPole-v1', render_mode='human')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.episodes = episodes
        self.target_update_frequency = target_update_frequency
        
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=2000,
            batch_size=32
        )
        
        self.scores = []
        self.episode_rewards = []
        self.epsilon_history = []
        
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('CartPole Training Metrics - Real-time', fontsize=14)
        
        self.ax1 = self.axes[0, 0]
        self.ax2 = self.axes[0, 1]
        self.ax3 = self.axes[1, 0]
        self.ax4 = self.axes[1, 1]
        
        self.score_line = None
        self.reward_line = None
        self.avg_score_line = None
        self.epsilon_line = None
    
    def _update_plots(self):
        episodes = range(1, len(self.scores) + 1)
        
        self.ax1.clear()
        self.ax1.plot(episodes, self.scores, 'b-', alpha=0.6, linewidth=0.8)
        self.ax1.set_title('Episode Scores')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Score')
        self.ax1.grid(True, alpha=0.3)
        
        self.ax2.clear()
        self.ax2.plot(episodes, self.episode_rewards, 'g-', alpha=0.6, linewidth=0.8)
        self.ax2.set_title('Episode Rewards')
        self.ax2.set_xlabel('Episode')
        self.ax2.set_ylabel('Reward')
        self.ax2.grid(True, alpha=0.3)
        
        self.ax3.clear()
        window = 100
        moving_avg = []
        for i in range(len(self.scores)):
            if i < window:
                moving_avg.append(np.mean(self.scores[:i+1]))
            else:
                moving_avg.append(np.mean(self.scores[i-window+1:i+1]))
        self.ax3.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
        if len(self.scores) >= 10:
            moving_avg_10 = []
            for i in range(len(self.scores)):
                if i < 10:
                    moving_avg_10.append(np.mean(self.scores[:i+1]))
                else:
                    moving_avg_10.append(np.mean(self.scores[i-9:i+1]))
            self.ax3.plot(episodes, moving_avg_10, 'orange', linewidth=1.5, alpha=0.7, label='Moving Avg (10)')
        self.ax3.set_title('Moving Average Scores')
        self.ax3.set_xlabel('Episode')
        self.ax3.set_ylabel('Average Score')
        self.ax3.legend()
        self.ax3.grid(True, alpha=0.3)
        
        self.ax4.clear()
        self.ax4.plot(episodes, self.epsilon_history, 'purple', linewidth=1.5)
        self.ax4.set_title('Epsilon (Exploration Rate)')
        self.ax4.set_xlabel('Episode')
        self.ax4.set_ylabel('Epsilon')
        self.ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def train(self):
        for episode in range(self.episodes):
            state, _ = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            steps = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state, [1, self.state_size])
                
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
                
                self.agent.replay()
            
            self.scores.append(steps)
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.agent.epsilon)
            
            if (episode + 1) % self.target_update_frequency == 0:
                self.agent.update_target_model()
            
            avg_score_10 = np.mean(self.scores[-10:]) if len(self.scores) >= 10 else np.mean(self.scores)
            avg_score_100 = np.mean(self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
            memory_size = len(self.agent.memory)
            
            print(f"Episode {episode + 1}/{self.episodes} | "
                  f"Score: {steps:4d} | "
                  f"Reward: {total_reward:6.2f} | "
                  f"Avg(10): {avg_score_10:6.2f} | "
                  f"Avg(100): {avg_score_100:6.2f} | "
                  f"Epsilon: {self.agent.epsilon:.3f} | "
                  f"Memory: {memory_size:4d}")
            
            self._update_plots()
        
        plt.ioff()
        self.env.close()
        return self.scores, self.episode_rewards
    
    def plot_results(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.scores)
        plt.title('Episode Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        
        plt.subplot(1, 2, 2)
        moving_avg = []
        window = 100
        for i in range(len(self.scores)):
            if i < window:
                moving_avg.append(np.mean(self.scores[:i+1]))
            else:
                moving_avg.append(np.mean(self.scores[i-window+1:i+1]))
        plt.plot(moving_avg)
        plt.title(f'Moving Average Score (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Score')
        
        plt.tight_layout()
        plt.savefig('cartpole_training_results.png')
        print("Training results saved to cartpole_training_results.png")
    
    def save_model(self, filepath='models/cartpole_model.h5'):
        self.agent.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/cartpole_model.h5'):
        self.agent.load(filepath)
        print(f"Model loaded from {filepath}")

