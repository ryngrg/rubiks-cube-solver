from cube import cube
import random

class agent():
    q_network = network()
    
    def __init__(self, epsilon = 0.01, train = True):
        self.train = train
        self.epsilon = epsilon
    
    def random_action(self):
        face = random.choice(['r', 'b', 'w', 'g', 'o', 'y'])
        clockwise = random.choice([True, False])
        return (face, clockwise)

    def greedy_action(self, state):
        best_q = None
        for face in ['r', 'b', 'w', 'g', 'o', 'y']:
            for clockwise in [True, False]:
                q = get_q(state, (face, clockwise))
                if (best_q is None) or (q > best_q):
                    best_q = q
                    best_action = (face, clockwise)
        return best_action

    def pick_action(self, state):
        if not(self.train) or (random.uniform(0, 1) > self.epsilon):
            return greedy_action(state)
        else:
            return random_action()

class network():
    pass
