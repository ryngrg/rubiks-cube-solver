from cube import Cube
import random
import keras
import numpy as np


def random_action():
    face = random.choice(['r', 'b', 'w', 'g', 'o', 'y'])
    clockwise = random.choice([True, False])
    return face, clockwise


class Network:
    def __init__(self):
        self.model = keras.model.Sequential()
        self.model.add(keras.layers.Dense())

    def state_to_input(self, state):
        pass

    def get_q(self, state):
        inputs = self.state_to_input(state)
        return self.model(inputs)


class Agent:
    q_network = Network()
    
    def __init__(self, epsilon=0.01, train=True):
        self.train = train
        self.epsilon = epsilon
        self.actions = []
        for face in ['r', 'b', 'w', 'g', 'o', 'y']:
            for clockwise in [True, False]:
                self.actions.append((face, clockwise))
        self.actions = tuple(self.actions)

    def greedy_action(self, state):
        q = self.q_network.get_q(state)
        return self.actions[np.argmax(q)]

    def pick_action(self, state):
        if not self.train or (random.uniform(0, 1) > self.epsilon):
            return self.greedy_action(state)
        else:
            return random_action()
