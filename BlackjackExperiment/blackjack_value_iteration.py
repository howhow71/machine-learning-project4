import numpy as np
from ValueIteration.value_iteration import ValueIteration
from mdp.blackjack.blackjack import BlackJackMDP
np.set_printoptions(precision=2)

blackjack_mdp = BlackJackMDP()
value_iteration_blackjack = ValueIteration(blackjack_mdp,max_iter=10,bellman_tolerance=0.01)

policy = value_iteration_blackjack.find_optimal_policy()
#print(value_iteration_blackjack.bellman_error)
#print(value_iteration_blackjack.bellman_error_history)
#print(len(blackjack_mdp.value))
#print(policy)
print(policy)
