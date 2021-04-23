import math
import copy
from pprint import pprint
import logging

class Planning():
    '''
    @author ashish juneja
    @author jai kumar
    Planning type problems are ones where the environment dynamics are known and learning happens through
    repeated updates to the state value function Value function V(s). Policy Iteration and Value Iteration are algorithms which use planning
    to solved a Markov Decision Process.
    '''
    def __init__(self, mdp, max_iter, bellman_tolerance):
      self.mdp = mdp
      self.max_iter = max_iter
      self.bellman_tolerance = bellman_tolerance
      self.bellman_error = math.inf

    def find_greedy_actions(self, state, new_value_function):
        '''Returns actions that maximizes value based on current policy'''
        max_actions = [(None, -math.inf)]
        for action in self.mdp.actions:
            q = 0
            new_states = self.mdp.actions[action](state)
            for state_prime in new_states:
                logging.info("Computing Value of State Moving from State %s to: %s with action %s with probability: %s", state, state_prime,action, self.mdp.transition_probability(state, action, state_prime))
                new_value = None
                if(isinstance(state_prime,tuple)):
                    new_value = new_value_function[state_prime[0]][state_prime[1]]
                else:
                    new_value = new_value_function[state_prime]
                    if(state_prime in self.mdp.terminal_states):
                        new_value = self.mdp.reward(state,action,state_prime)

                '''Update action value for each state by multiplying
                   the long term value of each successor state by the probability of ending up in that future state'''
                q += new_value * self.mdp.transition_probability(
                        state, action, state_prime)

            #print(q)
            #input()
            '''Act greedily, by selection actions that maximize q'''
            if (q > max_actions[0][1]):
                max_actions = [(action, q)]
            elif (q == max_actions[0][1]):
                max_actions.append((action, q))
        return max_actions

    def find_greedy_policy(self, new_value_function):
        '''Iterates through all states, and updates the policy with the action
           that maximizes value. '''
        new_policy = {}
        for state in self.mdp.states:
            if (state in self.mdp.terminal_states):
                continue
            new_action_qs = {}
            greedy_action = self.find_greedy_actions(state, new_value_function)
            for curr_max_action in greedy_action:
                new_action_qs[curr_max_action[0]] = 1 / len(greedy_action)
            new_policy[state] = new_action_qs
        return new_policy