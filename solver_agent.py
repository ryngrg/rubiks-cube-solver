from cube import cube
import random

def random_action():
    face = random.choice(['r', 'b', 'w', 'g', 'o', 'y'])
    clockwise = random.choice([True, False])
    return (face, clockwise)

def greedy_action(state):
    best_q = None
    for face in ['r', 'b', 'w', 'g', 'o', 'y']:
        for clockwise in [True, False]:
            q = get_q(state, (face, clockwise))
            if (best_q is None) or (q > best_q):
                best_q = q
                best_action = (face, clockwise)
    return best_action                

def get_q(state, action)
