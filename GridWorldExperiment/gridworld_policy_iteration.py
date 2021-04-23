from PolicyIteration.policy_iteration import PolicyIteration
from mdp.gridworld.grid_world_mdp import GridWorldMDP
import logging
import math
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def test_evaluate_policy():
    mdp = GridWorldMDP(terminal_states=[(0,0),(49,49)],grid_size=50)
    expected_policy = {(0, 1): {'UP': 1.0}, (1, 2): {'UP': 0.5, 'LEFT': 0.5}, (3, 2): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (1, 3): {'UP': 0.5, 'LEFT': 0.5}, (3, 3): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (3, 0): {'LEFT': 1.0}, (3, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 0): {'LEFT': 1.0}, (1, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 3): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (2, 2): {'UP': 0.5, 'LEFT': 0.5}, (1, 0): {'LEFT': 1.0}, (0, 2): {'UP': 1.0}, (0, 3): {'UP': 1.0}}
    grid_world_pi = PolicyIteration(mdp, max_iter=100,bellman_tolerance=0)
    optimal_policy = grid_world_pi.find_optimal_policy()
    #logging.info("Found Policy : %s",optimal_policy)
    print(len(grid_world_pi.bellman_error_history))
    print(grid_world_pi.metrics)

    plt.plot(grid_world_pi.bellman_error_history)
    #plt.hold(True)
    #plt.plot(pred_upd)
    plt.show()
    plt.savefig("policy iteration.png")

test_evaluate_policy()

