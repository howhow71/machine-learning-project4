import unittest
from ValueIteration.value_iteration import ValueIteration
from mdp.gridworld.grid_world_mdp import GridWorldMDP
import logging
logging.basicConfig(level=logging.INFO)

class TestValueIteration(unittest.TestCase):
  def test_evaluate_policy(self):
    mdp = GridWorldMDP(terminal_states=[(0,0)])
    expected_policy = {(0, 1): {'UP': 1.0}, (1, 2): {'UP': 0.5, 'LEFT': 0.5}, (3, 2): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (1, 3): {'UP': 0.5, 'LEFT': 0.5}, (3, 3): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (3, 0): {'LEFT': 1.0}, (3, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 0): {'LEFT': 1.0}, (1, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 3): {'UP': 0.25, 'DOWN': 0.25, 'LEFT': 0.25, 'RIGHT': 0.25}, (2, 2): {'UP': 0.5, 'LEFT': 0.5}, (1, 0): {'LEFT': 1.0}, (0, 2): {'UP': 1.0}, (0, 3): {'UP': 1.0}}
    optimal_policy = ValueIteration(mdp, max_iter=3,
                    bellman_tolerance=0.01).find_optimal_policy()
    logging.info("Found Policy : %s",optimal_policy)
    self.assertEqual(expected_policy, optimal_policy)

def main():
  unittest.main()

if __name__ == "__main__":
  main()