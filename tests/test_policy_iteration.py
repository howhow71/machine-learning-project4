import unittest
from PolicyIteration.policy_iteration import PolicyIteration
from mdp.gridworld.grid_world_mdp import GridWorldMDP
import logging
logging.basicConfig(level=logging.INFO)
class TestPolicyIteration(unittest.TestCase):
  def test_evaluate_policy(self):
    mdp = GridWorldMDP(terminal_states=[(0,0),(3,3)])
    expected_policy = {(0, 1): {'UP': 1.0}, (1, 2): {'UP': 0.5, 'RIGHT': 0.5}, (3, 2): {'DOWN': 1.0}, (1, 3): {'RIGHT': 1.0}, (3, 0): {'DOWN': 0.5, 'LEFT': 0.5}, (3, 1): {'DOWN': 1.0}, (2, 1): {'DOWN': 0.5, 'LEFT': 0.5}, (2, 0): {'LEFT': 1.0}, (1, 1): {'UP': 0.5, 'LEFT': 0.5}, (2, 3): {'RIGHT': 1.0}, (2, 2): {'DOWN': 0.5, 'RIGHT': 0.5}, (1, 0): {'LEFT': 1.0}, (0, 2): {'UP': 1.0}, (0, 3): {'UP': 0.5, 'RIGHT': 0.5}}
    optimal_policy = PolicyIteration(mdp, max_iter=3,
                    bellman_tolerance=0.01).find_optimal_policy()
    logging.info("Found Policy : %s",optimal_policy)
    self.assertEqual(expected_policy, optimal_policy)

def main():
  unittest.main()

if __name__ == "__main__":
  main()