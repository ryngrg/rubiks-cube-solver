from cube import Cube
import random
from solver_agent import Agent
from solver_agent import state_to_input


def get_shuffled_cube(num_moves):
    shuffled_cube = Cube()
    for i in range(num_moves):
        face, clockwise = random_action()
        shuffled_cube.turn_face(face, clockwise)
    return shuffled_cube


def random_action():
    face = random.choice(('r', 'b', 'w', 'g', 'o', 'y'))
    clockwise = random.choice((True, False))
    return face, clockwise


if __name__ == "__main__":
    rc = get_shuffled_cube(10)
    rc.display()
    agent = Agent()
    inputs = state_to_input(rc.get_state())
    print(inputs.shape)
    print(inputs)
