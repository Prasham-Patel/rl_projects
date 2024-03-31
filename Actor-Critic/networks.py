import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # removes TF warning
import tensorflow.python.keras as keras
from tensorflow.python.keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    def __init__(self, actions, Layers = [1024, 512], name = "actor-critic", chkpt_dir = "tmp/actor_critic"):
        super(ActorCriticNetwork, self).__init__()
        self.Layer1_dim = Layers[0]
        self.Layer2_dim = Layers[1]
        self.actions = actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.file = os.path.join(self.checkpoint_dir, name+"ac")
        
        self.Layer1 = Dense(self.Layer1_dim, activation= 'relu')
        self.Layer2 = Dense(self.Layer2_dim, activation= 'relu')
        self.value = Dense(1, activation= None)
        self.policy = Dense(actions, activation="softmax")
        
    def call(self, state):
        value = self.Layer1(state)
        value = self.Layer2(value)   
        v = self.value(value)
        pi = self.policy(value)
        return v, pi