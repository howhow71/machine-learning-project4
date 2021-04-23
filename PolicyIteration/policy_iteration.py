from mdp.gridworld.grid_world_mdp import GridWorldMDP
from Planning.planning import Planning
import math
import copy
from pprint import pprint
import numpy as np
import logging
np.set_printoptions(precision=2,suppress= True)


class PolicyIteration(Planning):
    '''
        Implementation of Policy Iteration Reinforcement Learning Algorithm.
        Used for Planning Problems - when full dynamics of a system are known.
        @author ashish juneja
        @author jai kumar
    '''
    def __init__(self, mdp, max_iter, bellman_tolerance):
        super().__init__(mdp, max_iter, bellman_tolerance)
        self.bellman_error_history = []
        self.metrics = {}

    def one_step_lookahead(self, state):
        '''Iterates through all actions of a given state and returns an expectation of the updated
        value function for an input State, using Bellman Expectation Equation'''
        updated_value = 0
        for action in self.mdp.actions:
            '''Invoke an action, and determine a list of possible new states we can end up in'''
            possible_new_states = self.mdp.actions[action](state)
            for new_state in possible_new_states:
                '''Compute the value of the new state by using the Bellman Expectation Equation'''
                value_of_new_state = self.mdp.value[new_state[0]][new_state[1]]
                updated_value += self.mdp.policy[state][action] * \
                    self.mdp.transition_probability(
                        state, action, new_state)*(self.mdp.reward(new_state, action)+value_of_new_state)

        return updated_value

    def update_value_function(self):
        ''' Iterates through all states in MDP and updates the value function
            with a one step lookahead
        '''
        current_policy = self.mdp.policy
        current_value_function = copy.deepcopy(self.mdp.value)
        for state in self.mdp.states:
            if (state in self.mdp.terminal_states):
                continue
            new_value = self.one_step_lookahead(state)
            current_value_function[state[0]][state[1]] = new_value
        return current_value_function


    def find_optimal_policy(self):
        old_policy = None
        for i in range(0,self.max_iter):
            logging.info("Policy Iteration %s",i)
            self.metrics[i] = []
            new_value_function = self.evaluate_policy(i)
            new_policy = self.find_greedy_policy(new_value_function)
            if(new_policy == old_policy):
                logging.info("Policy Converged After %i iterations. Exiting Early.",i)
                break
            else:
                old_policy = new_policy
        return new_policy

    def evaluate_policy(self,iter):
        self.metrics[iter].append(len(self.bellman_error_history))
        for i in range(0, self.max_iter):
            logging.info("Iteration #{} in evaulate policy: ".format(str(i)))
            old_value_function = self.mdp.value
            new_value_function = self.update_value_function()
            bellman_error = abs(
                np.max((np.subtract(old_value_function, new_value_function))))
            self.mdp.value = new_value_function


            #logging.info("MDP Value Function After Iteration {}:\n {}".format(i, self.mdp))
            self.bellman_error_history.append(bellman_error)
            if (bellman_error <= self.bellman_tolerance):
                logging.info("exiting early, bellman error is below tolerance")
                break

        return new_value_function


if __name__ == "__main__":
    mdp = GridWorldMDP(terminal_states=[(0,0),(3,3)])
    PolicyIteration(mdp, max_iter=3,
                    bellman_tolerance=0.01).find_optimal_policy()
