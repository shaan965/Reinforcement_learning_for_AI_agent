import numpy as np

from action_value_table import ActionValueTable

# HINT: You will need to sample from probability distributions to complete this
# question.  Please use `numpy.random.rand` to generate a floating point number
# uniformly at random and `numpy.random.choice` to uniformly randomly select an
# element from a list.
#
# Documentation:
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3  # agents actions
GAMMA = 0.95
STEP_SIZE = 0.25
EPSILON = 0.1


class QLearningAgent():
    """
    Implement your code for a Q-learning agent here. We have provided code
    implementing the action-value table in `action_value_table.py`. Here, you
    will implement the `get_action`, `get_greedy_action` and `update` methods.
    """

    def __init__(self, dimension):
        self.actions = [UP, DOWN, LEFT, RIGHT]
        num_actions = len(self.actions)
        self.values = ActionValueTable(dimension, num_actions)
        self.gamma = GAMMA
        self.step_size = STEP_SIZE
        self.epsilon = EPSILON


    def update(self, state, action, reward, next_state, done):
        """
        Update the values stored in `self.values` using Q-learning.

        HINT: Use `self.values.get_value` and `self.values.set_value`
        HINT: Remember to add a special case to handle the terminal state

        Args:
            state : (list) a list of type [bool, int, int] where the first
            entry is whether the agent possesses the key, and the next two
            entries are the row and column position of the agent in the maze
            
            action : (int) the action taken at state

            reward : float

        Returns:
            None
        """

        if done:
            # Q(terminal,.)=0
            new_val = self.values.get_value(state,action) + (self.step_size*(reward + ((self.gamma * 0) - self.values.get_value(state,action))))
            self.values.set_value(state,action,new_val)
        else:
            new_val = self.values.get_value(state,action) + (self.step_size*(reward + ((self.gamma * self.values.get_value(next_state,self.get_greedy_action(next_state))) - self.values.get_value(state,action))))
            self.values.set_value(state,action,new_val)




    def get_action(self, state):
        """
        This function returns an action from self.actions given a state. 

        Implement this function using an epsilon-greedy policy. 

        HINT: use np.random.rand() to generate random numbers
        HINT: If more than one action has maximum value, treat them all as the
        greedy action. In other words, if there are b greedy actions, each
        should have probability
        (1 - epsilon)/b + epsilon/|A|,
        where |A| is the number of actions in this state.

        Args:
            state : (list)
            a list of type [bool, int, int] where the first entry is whether
            the agent posseses the key, and the next two entries are the row
            and column position of the agent in the maze

        Returns:
            action : (int) a epsilon-greedy action for `state`
        """

        # exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)

        # exploitation
        else:
            return self.get_greedy_action(state)



    def get_greedy_action(self, state):
        """
        This function returns an action from self.actions given a state. 

        Implement this function using a greedy policy, i.e. return the action
        with the highest value
        
        HINT: If more than more than one action has maximum value, uniformly
        randomize amongst them

        Args:
            state : (list)
            a list of type [bool, int, int] where the first entry is whether
            the agent posseses the key, and the next two entries are the row
            and column position of the agent in the maze

        Returns:
            action : (int) a greedy action for `state`
        """
        max_value = float("-inf")
        max_actions = []
        for a in self.actions:
            val = self.values.get_value(state, a)
            if val > max_value:
                max_value = val
                max_actions = []
                max_actions.append(a)
            elif val == max_value:
                max_actions.append(a)

        action = np.random.choice(max_actions)

        return action

