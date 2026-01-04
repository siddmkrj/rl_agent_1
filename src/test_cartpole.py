import gymnasium as gym
import numpy as np
from cartpole_trainer import CartPoleTrainer


def test_trained_agent(model_path='models/cartpole_model.h5', num_episodes=10, render=True):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    trainer = CartPoleTrainer(episodes=1)
    trainer.load_model(model_path)
    
    total_scores = []
    
    original_epsilon = trainer.agent.epsilon
    trainer.agent.epsilon = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, trainer.state_size])
        total_reward = 0
        steps = 0
        
        while True:
            action = trainer.agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, trainer.state_size])
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        total_scores.append(steps)
        print(f"Episode {episode + 1}: Score = {steps}, Reward = {total_reward}")
    
    env.close()
    
    trainer.agent.epsilon = original_epsilon
    
    print(f"\nTest Results:")
    print(f"Average Score: {np.mean(total_scores):.2f}")
    print(f"Best Score: {max(total_scores)}")
    print(f"Worst Score: {min(total_scores)}")


if __name__ == "__main__":
    test_trained_agent(num_episodes=10, render=True)

