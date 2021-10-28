import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

colors = ('r', 'b', 'w', 'g', 'o', 'y')


def state_to_input(state):
    state_list = np.array([colors.index(i) for face in state.values() for i in face])
    state_ohvs = tf.keras.utils.to_categorical(state_list, num_classes=6)

    return np.reshape(state_ohvs, (48*6))


class Network:
    def __init__(self, model=None):
        if model is not None:
            self.model = tf.keras.models.load_model(r"./models/" + model)
        else:
            self.model = Sequential()
            self.model.add(Dense(256, activation='relu', input_shape=(1, 48 * 6)))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(144, activation='relu'))
            self.model.add(Dense(144, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(96, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(48, activation='relu'))
            self.model.add(Dense(24, activation='relu'))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dense(12))

            opt = tf.keras.optimizers.Adam(learning_rate=0.01)
            self.model.compile(loss="mse", optimizer=opt)
            self.model.summary()

    def update_weights(self, X, Y):
        pass

    def get_q(self, state, train=True):
        inputs = state_to_input(state)
        return self.model(inputs, training=train)


class Agent:
    def __init__(self, init_epsilon=0.01, eps_episode_decay=0.9, eps_time_decay=0.9, model=None):
        self.train = True
        self.init_epsilon = init_epsilon
        self.eps_episode_decay = eps_episode_decay
        self.eps_time_decay = eps_time_decay
        self.actions = []
        for face in colors:
            for clockwise in (True, False):
                self.actions.append((face, clockwise))
        self.actions = tuple(self.actions)
        self.q_network = Network(model)

    def learn(self, cube, num_episodes, t_max):
        self.train = True
        og_state = cube.get_state()
        first_epsilon = self.init_epsilon
        for n in range(num_episodes):
            N = np.zeros((1, 12))
            for t in range(t_max):
                epsilon = first_epsilon
                state = cube.get_state()
                q = self.q_network.get_q(state, True)
                a = self.pick_action(state, epsilon)
                cube.turn_face(self.actions[a][0], self.actions[a][1])
                N[0][a] += 1
                q_new = q + (cube.state_reward() - q) / N
                self.q_network.update_weights(state, q_new)
                if cube.is_solved():
                    break
                epsilon *= self.eps_time_decay
            cube.set_state(og_state)
            first_epsilon *= self.eps_episode_decay

    def perform(self, cube, max_turns):
        self.train = False
        turns = 0
        while (turns < max_turns) and not(cube.is_solved()):
            a = self.pick_action(cube.get_state())
            cube.turn_face(self.actions[a][0], self.actions[a][1])
            turns += 1
            print("Turn", turns, ":", self.actions[a])
        return cube.is_solved(), turns

    def greedy_action(self, state):
        q = self.q_network.get_q(state, self.train)
        return np.argmax(q)

    def pick_action(self, state, epsilon=0.0):
        if not self.train or (random.uniform(0, 1) > epsilon):
            return self.greedy_action(state)
        else:
            return random.randint(0, len(self.actions)-1)

    def save_model(self, name):
        self.q_network.model.save(r"./models/" + name)
