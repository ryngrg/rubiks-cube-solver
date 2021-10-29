from cube import Cube
import random
from solver_agent import Agent


def random_action():
    face = random.choice(('r', 'b', 'w', 'g', 'o', 'y'))
    clockwise = random.choice((True, False))
    return face, clockwise


def get_shuffled_cube(num_moves):
    shuffled_cube = Cube()
    for i in range(num_moves):
        face, clockwise = random_action()
        shuffled_cube.turn_face(face, clockwise)
    return shuffled_cube


def train_main(model_name=None, render=False):
    agent = Agent(model=model_name)
    for moves in range(1, 21):
        for i in range(5):
            cube = get_shuffled_cube(moves)
            if render:
                cube.display()
            agent.learn(cube, 3, 40)
        agent.save_model("trainedModel_" + str(moves) + "_moves")


def test_main(model_name, num_moves, render=False):
    agent = Agent(model=model_name)
    rc = get_shuffled_cube(num_moves)
    successful, moves = agent.perform(rc, 40, render)
    print("Successful:", successful)
    print("Solved in moves:", moves)


if __name__ == "__main__":
    train_main()
    test_main("trainedModel_20_moves", 15)
