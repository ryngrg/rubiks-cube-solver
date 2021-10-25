from cube import Cube
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

colors = ('r', 'b', 'w', 'g', 'y', 'o')


def random_action():
    face = random.choice(colors)
    clockwise = random.choice([True, False])
    return face, clockwise


def state_to_input(state):
    state_list = np.array([colors.index(i) for face in state.values() for i in face])
    print(state_list)
    return tf.keras.utils.to_categorical(state_list, num_classes=6)


class Network:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Dense(36, activation='relu', input_shape=(1, 48, 6)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(12))

        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss="mse", optimizer=opt)
        self.model.summary()

    def get_q(self, state, train=True):
        inputs = state_to_input(state)
        return self.model(inputs, training=train)


class Agent:
    q_network = Network()

    def __init__(self, epsilon=0.01, train=True):
        self.train = train
        self.epsilon = epsilon
        self.actions = []
        for face in colors:
            for clockwise in [True, False]:
                self.actions.append((face, clockwise))
        self.actions = tuple(self.actions)

    def greedy_action(self, state):
        q = self.q_network.get_q(state, self.train)
        return self.actions[np.argmax(q)]

    def pick_action(self, state):
        if not self.train or (random.uniform(0, 1) > self.epsilon):
            return self.greedy_action(state)
        else:
            return random_action()
