from mdp.gridworld.grid_world_mdp import GridWorldMDP
from Planning.planning import Planning
import math
import copy
from pprint import pprint
import numpy as np
import logging
np.set_printoptions(precision=2)

class ValueIteration(Planning):
    def __init__(self, mdp, max_iter, bellman_tolerance):
        super().__init__(mdp, max_iter, bellman_tolerance)
        self.bellman_error_history = []

    def one_step_lookhead(self, state):
        '''Returns an updated value of the state based on the Bellman Optimality Equation'''
        max_action_value = float("-inf")
        for action in self.mdp.actions:
            updated_value = 0
            possible_new_states = self.mdp.actions[action](state)
            #print("taking action " + action + " from state " + str(state))
            #print("new states " + str(possible_new_states))



            '''Updates the value of the current state to the value of successive
            states * transition probability'''
            for new_state in possible_new_states:
                value_of_new_state = self.mdp.get_value([new_state]) #self.mdp.value[new_state[0]][new_state[1]]
                reward = self.mdp.reward(state, action,new_state)

                #logging.info("Probability of Transition from %s to: %s = %s with action %s, reweard=%s", state, new_state, self.mdp.transition_probability(
                #    state, action, new_state), action, reward)
                #print("reward " + str(self.mdp.reward(state, action)))
                #if(state == 19):
                #    input()


                updated_value += self.mdp.transition_probability(
                    state, action, new_state) * (value_of_new_state + self.mdp.reward(state, action,new_state))
                #print("updated value" + str(updated_value))
                #print("reward " + str(self.mdp.reward(state, action)))
                #input()
                '''Adds immediate reward for action taken'''
            #updated_value += self.mdp.reward(state, action)
            #print(updated_value)

            if (updated_value > max_action_value):
                max_action_value = updated_value
        return max_action_value

    def find_optimal_policy(self):
      curr_iter = 0
      while (self.max_iter >= curr_iter and self.bellman_error > self.bellman_tolerance ):
        logging.info("Entering iteration %i ",curr_iter)
        curr_iter +=1
        new_value_function = self.update_value_function()
        self.bellman_error = np.max(abs((np.subtract(self.mdp.value, new_value_function))))
        sum_squared_error = np.sum(abs(np.subtract(self.mdp.value, new_value_function)))
        self.bellman_error_history.append(sum_squared_error)
        self.mdp.value = new_value_function
        logging.info("MDP Value Function After Iteration {}:\n {}".format(curr_iter, np.around(np.array(self.mdp.value),2)))
      final_policy = self.find_greedy_policy(self.mdp.value)
      return final_policy


    def update_value_function(self):
        '''Applies Bellman Optimality Equation to every state and returns the updated value'''
        current_value_function = copy.deepcopy(self.mdp.value)
        for state in self.mdp.states:
            if (state in self.mdp.terminal_states):
                continue
            new_value = self.one_step_lookhead(state)
            if(isinstance(state,tuple)):
                current_value_function[state[0]][state[1]] = new_value
            else:
                current_value_function[state] = new_value
        return current_value_function


if __name__ == "__main__":
    mdp = GridWorldMDP(terminal_states=[(0, 0)])
    ValueIteration(mdp, max_iter=3, bellman_tolerance=0.01).find_optimal_policy()
