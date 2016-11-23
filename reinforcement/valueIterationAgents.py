# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for iteration in range(iterations):
            # make a copy so that we do "batch" updates
            # rather than "online" updates (see instructions)
            old_values = self.values.copy()

            # for each state in our MDP...
            for state in mdp.getStates():
                q_values = util.Counter()
                # loop over all possible actions we can take from this state
                for action in mdp.getPossibleActions(state):
                    # a list of (new_state, prob) tuples
                    transitions = mdp.getTransitionStatesAndProbs(state, action)
                    first_term = util.Counter()
                    second_term = util.Counter()
                    for new_state, prob in transitions:
                        reward = mdp.getReward(state, action, new_state)
                        first_term[new_state] = prob*reward
                        second_term[new_state] = prob*self.discount*old_values[new_state]
                    # sum over new states
                    q_values[action] = (first_term + second_term).totalCount()

                # get the best action
                best_action = q_values.argMax()
                self.values[state] = q_values[best_action]

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # a list of (new_state, prob) tuples
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        first_term = util.Counter()
        second_term = util.Counter()
        for new_state, prob in transitions:
            reward = self.mdp.getReward(state, action, new_state)
            first_term[new_state] = prob*reward
            second_term[new_state] = prob*self.discount*self.values[new_state]

        # sum over new states
        q_value = (first_term + second_term).totalCount()
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        q_values = util.Counter()
        # loop over all possible actions we can take from this state
        for action in self.mdp.getPossibleActions(state):
            q_values[action] = self.computeQValueFromValues(state, action)
                
        # get the best action
        best_action = q_values.argMax()
        if best_action and len(best_action):
            return best_action
        else:
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
