from pprint import pprint


class GridWorldMDP:
    '''
        Sample 4x4 Grid world as described in David Silvers 3rd RL Lecture.
        Consists of self.grid_size actions, UP,DOWN,LEFT,RIGHT. Fully deterministic, transition probability is 1.0 for an action (i.e. UP will always
        move up.). Default initial policy is uniform random.
        @author ashish juneja
        @author jai kumar
    '''
    def __init__(self,
                 initial_value=None,
                 initial_policy=None,
                 terminal_states=[(0, 0)],
                 grid_size=4):
        self.grid_size = grid_size
        self.init_states()
        self.terminal_states = terminal_states
        self.actions = {
            'UP': self.up,
            'DOWN': self.down,
            'LEFT': self.left,
            'RIGHT': self.right
        }

        if (not initial_value):
            self.value = [[0] * self.grid_size for _ in range(self.grid_size)]
        if (not initial_policy):
            self.policy = self.generate_initial_policy()

    def get_value(self,state):
        state = state[0]
        return self.value [state[0]][state[1]]

    def init_states(self):
        self.states = set()
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.states.add((x, y))

    def generate_initial_policy(self):
        initial_policy = {}
        initial_states = [[0] * self.grid_size] * self.grid_size
        for index_x, state_x in enumerate(initial_states):
            for index_y, state_y in enumerate(initial_states):
                curr_state = (index_x, index_y)
                if (curr_state in self.terminal_states):
                    continue
                for action in self.actions:
                    if (curr_state not in initial_policy):
                        initial_policy[curr_state] = {action: 0.25}
                    else:
                        initial_policy[curr_state][action] = 0.25
        return initial_policy

    def transition_probability(self, state, action, next_state):
        final_state = self.actions[action](state)[0]
        return 1 if final_state == next_state else 0

    def reward(self, state, action):
        return -1

    def up(self, state):
        next_step = (state[0], state[1] - 1)
        if (state in self.terminal_states
                or not self.in_bounds(next_step[0], next_step[1])):
            return [state]
        else:
            return [next_step]

    def down(self, state):
        next_step = (state[0], state[1] + 1)
        if (state in self.terminal_states
                or not self.in_bounds(next_step[0], next_step[1])):
            return [state]
        else:
            return [next_step]

    def left(self, state):
        next_step = (state[0] - 1, state[1])
        if (state in self.terminal_states
                or not self.in_bounds(next_step[0], next_step[1])):
            return [state]
        else:
            return [next_step]

    def right(self, state):
        next_step = (state[0] + 1, state[1])
        if (state is self.terminal_states
                or not self.in_bounds(next_step[0], next_step[1])):
            return [state]
        else:
            return [next_step]

    def in_bounds(self, x, y):
        return x in range(0, self.grid_size) and y in range(0, self.grid_size)

    def __repr__(self):
        matrix = ""
        for row in self.value:
            for value in row:
                matrix += '{:0.2f}\t'.format(value)
            matrix += "\n"
        return matrix
