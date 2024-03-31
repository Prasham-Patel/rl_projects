import numpy as np
from numpy import float32
import tensorflow as tf
import tensorflow.python.keras as keras
import keras
# from keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticNetwork

class Agent:
    def __init__(self, alpha = 0.0003, gamma = 0.99, num_actions = 2):
        self.gamma = gamma
        self.num_actions = num_actions
        self.action = None
        self.action_space = [i for i in range(self.num_actions)]
        self.actor_critic = ActorCriticNetwork(num_actions)
        # opt = keras.optimizers.Adam(learning_rate=alpha)
        self.actor_critic.compile(optimizer= 'adam')
        
    def choose_action(self, observation):
        # print([observation].shape)
        
        state = tf.convert_to_tensor(np.asarray([observation], dtype=float32))
        _, probs = self.actor_critic(state)
        action_probability = tfp.distributions.Categorical(probs)
        actions = action_probability.sample()
        self.actions = actions
        
        return self.actions.numpy()[0]
    
    def save_model(self):
        print("....saving model....")
        self.actor_critic.save_weights(self.actor_critic.checkpoint_dir)
    
    def load_model(self):
        print("...loading model...")
        self.actor_critic.load_weights(self.actor_critic.checkpoint_dir)
        
    def learn(self, state, reward, state_next, done):
        state = tf.convert_to_tensor([state], dtype=float32)
        state_next = tf.convert_to_tensor([state_next], dtype=float32)
        reward = tf.convert_to_tensor([reward], dtype=float32)
        
        with tf.GradientTape() as tape:
            state_value, probs = self.actor_critic(state)
            state_value_next, _ = self.actor_critic(state_next)
            state_value = tf.squeeze(state_value)
            state_value_next = tf.squeeze(state_value_next)
                       
            action_probs = tfp.distributions.Categorical(probs)
            log_prob = action_probs.log_prob(self.actions)
            
            delta = reward + self.gamma*state_value_next*(1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta**2
            
            total_loss = actor_loss + critic_loss
            
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))
        
            