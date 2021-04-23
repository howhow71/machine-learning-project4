from mdp.blackjack.constants import TWENTY_ONE,OVER_TWENTY_ONE,KEEP
import logging
import random

logging.basicConfig(level=logging.INFO)
class BlackJackMDP:
    def __init__(self, initial_value=None,initial_policy=None,terminal_states=[(21,21)]):

        self.actions = {
            'DRAW': self.draw,
            'STOP': self.stop
        }
        self.states = set(range(0,44))
        self.cards = [*range(1,7)]
        self.terminal_states = [*range(22,44)]
        if (not initial_value):
            self.value = [0 for _ in range(len(self.states)+1)]

    def get_value(self,state):
        return self.value[state[0]]

    def draw(self, state):
        if(state in self.terminal_states):
            return []
        if(state > 21):
            return [43]
        possible_next_states = []
        for card in self.cards:
            if(state+card > 21):
                possible_next_states.append(43)
                break
            possible_next_states.append(state+card)
        #print("possible new states for: " + str(state) + ": " + str(possible_next_states))
        return possible_next_states

    def stop(self,state):
        if(state > 21):
            return [43]
        else:
            return [state + 22]

    def reward(self,state,action,new_state):
        #print("entering reward with " + str(state), ": " + str(action))
        if(state < 21 and action == "STOP"):
            return state
        if(new_state == 21):
            return 100
        if(new_state > 21):
            return  state - 22
        else:
            return 0
        # elif(new_state > 21 and KEEP):
        #     return 0
        # elif(action == "STOP" and new_state == KEEP):
        #     return state
        # elif (action == "DRAW"):
        #     return 0

    def is_terminal_state(self,state,action):
        pass

    def transition_probability(self, state, action, next_state):
        if(state >= 21 and next_state == 43):
            return 1
        elif(state >= 21 and next_state != 43):
            return 0

        if(action == "STOP" and next_state == state + 22):
            return 1
        elif(action == "STOP" and next_state != state + 22):
            return 0
        if(action == "DRAW"):
            if(next_state < state):
                return 0
            number_over_twenty_one = 0
            if(next_state in range(state,state+len(self.cards)+1)):
                    return 1/len(self.cards)
            else:
                return 0