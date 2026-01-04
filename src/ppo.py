import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4, gamma=0.99, 
                 epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        self.actor = self._build_actor(learning_rate)
        self.critic = self._build_critic(learning_rate)
        
    def _build_actor(self, learning_rate):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        
        mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        std = tf.keras.layers.Dense(self.action_dim, activation='softplus')(x)
        std = std + 1e-5
        
        model = tf.keras.Model(inputs=inputs, outputs=[mean, std])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        return model
    
    def _build_critic(self, learning_rate):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        value = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=value)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        
        return model
    
    def select_action(self, state, training=True):
        state = np.expand_dims(state, axis=0)
        mean, std = self.actor(state)
        
        if training:
            dist = tfp.distributions.Normal(mean, std)
            action = dist.sample()
            action = tf.clip_by_value(action, -1.0, 1.0)
        else:
            action = mean
        
        return action.numpy()[0]
    
    def get_value(self, state):
        state = np.expand_dims(state, axis=0)
        value = self.critic(state)
        return value.numpy()[0, 0]
    
    def compute_returns(self, rewards, dones, values, next_value=0):
        returns = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        last_gae = 0
        next_value = next_value
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_gae = delta + self.gamma * 0.95 * last_gae
            next_value = values[t]
        
        returns = advantages + values
        return returns, advantages
    
    def train(self, states, actions, old_log_probs, returns, advantages):
        states = np.array(states)
        actions = np.array(actions)
        old_log_probs = np.array(old_log_probs)
        returns = np.array(returns)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        with tf.GradientTape() as tape:
            mean, std = self.actor(states, training=True)
            dist = tfp.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions)
            log_probs = tf.reduce_sum(log_probs, axis=1)
            
            ratio = tf.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
            
            entropy = tf.reduce_mean(dist.entropy())
            actor_loss = actor_loss - self.entropy_coef * entropy
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        with tf.GradientTape() as tape:
            values = self.critic(states, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        
        return actor_loss.numpy(), critic_loss.numpy()
    
    def save(self, filepath):
        self.actor.save_weights(f"{filepath}_actor.h5")
        self.critic.save_weights(f"{filepath}_critic.h5")
    
    def load(self, filepath):
        self.actor.load_weights(f"{filepath}_actor.h5")
        self.critic.load_weights(f"{filepath}_critic.h5")

