from cube import Cube
from solver_agent import Agent
from solver_agent import random_action


def get_shuffled_cube(num_moves):
    shuffled_cube = Cube()
    for i in range(num_moves):
        shuffled_cube.turn_face(random_action()[0], random_action()[1])
    return shuffled_cube


if __name__ == "__main__":
    rc = get_shuffled_cube(10)
    rc.display()
