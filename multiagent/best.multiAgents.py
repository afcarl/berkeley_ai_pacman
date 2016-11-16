# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        #print "Getting scores..."
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentCapsules = currentGameState.getCapsules()
        newCapsules = successorGameState.getCapsules()
        newCapsuleDists = [manhattanDistance(capsule, newPos) for capsule in newCapsules]
        newFoodDists = [manhattanDistance(food, newPos) for food in newFood.asList()]
        currentGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        newGhostDists = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"

        # find the closest ghost
        closestGhost = min(newGhostDists)
        # if we've eaten a power pellet go after it
        if newScaredTimes[0] != 0:
            if closestGhost != 0:
                ghostTerm = 1./closestGhost
            else:
                ghostTerm = 1./0.0001
        # otherwise don't worry about it unless it's within
        # 4 steps of us, then take evasive action
        else:
            ghostTerm = min(4, closestGhost)

        # try to go after power pellets
        # add in a big bonuse for eating one
        if len(currentCapsules) > len(newCapsules):
            capsuleTerm = 50
        elif len(newCapsuleDists):
            closestCapsule = min(newCapsuleDists)
            if closestCapsule > 1:
                capsuleTerm = 1./closestCapsule
            else:
                capsuleTerm = 20
        else:
            capsuleTerm = 20

        # try to get closer to food
        if len(newFoodDists):
            foodTerm = 1./(sum(newFoodDists)/len(newFoodDists))
        else:
            foodTerm = 0

        return 1.5*successorGameState.getScore() + ghostTerm + foodTerm + capsuleTerm

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #print "Getting best action for state: \n" + str(gameState)
        print "Getting best action for pos: \n" + str(gameState.getPacmanPosition())
        numAgents = gameState.getNumAgents()
        print "We have " + str(numAgents) + " agents..."
        legalMoves = gameState.getLegalActions(0)
        max_value = float("-inf")
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(0, action)
            # get the value of the successor under control of the
            # next agent at the same depth as this step
            _, value = self.getValue(successorGameState, (0+1) % numAgents, depth=0)
            #print "value=" + str(value)
            if value > max_value:
                max_value = value
                max_action = action
        #print "Got an overall max value of " + str(max_value) + " for action " + str(max_action)
        #print "Returning " + str(max_action)
        return max_action

    # get the value of a given state, controlled by a
    # given agent at a given depth (number of moves down
    # the tree)
    def getValue(self, gameState, agentIndex, depth):
        print "getValue for state under control of agent " + str(agentIndex) + " at depth " + str(depth)
        # if we've gotten to our terminal depth just
        # return the evaluation function for that state
        numAgents = gameState.getNumAgents()
        if depth == self.depth:
            #print "reached max depth for agent " + str(agentIndex)
            return (None, self.evaluationFunction(gameState))

        if gameState.isWin() or gameState.isLose():
            return (None, self.evaluationFunction(gameState))

        # increment only when we see another pacman, not another ghost
        nextAgentIndex = (agentIndex+1) % numAgents
        print "nextAgentIndex is " + str(nextAgentIndex)
        if nextAgentIndex == 0:
            nextDepth = depth + 1
        else:
            nextDepth = depth

        # otherwise, if we're pacman...
        if agentIndex == 0:
            #print "Getting pacman max..."
            return self.getMaxValueOverSuccessors(gameState, nextAgentIndex, nextDepth)
        # ghost
        else:
            return self.getMinValueOverSuccessors(gameState, nextAgentIndex, nextDepth)

    # get maximum value over successors to gameState when those successors are
    # under the control of agentIndex
    def getMaxValueOverSuccessors(self, gameState, agentIndex, depth):
        print "Getting max value for states under control of agent " + str(agentIndex) + " at depth " + str(depth) #+ " in state:\n" + str(gameState) + "!!"
        max_value = float("-inf")
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            _, value = self.getValue(successorGameState, agentIndex, depth)
            #print "In max got value "+str(value)
            if value > max_value:
                max_value = value
                max_action = action

        #print "Max returning "+str(max_value)
        return (max_action, max_value)

    def getMinValueOverSuccessors(self, gameState, agentIndex, depth):
        print "Getting min value for states under control of agent " + str(agentIndex) + " at depth " + str(depth) #+ " in state:\n" + str(gameState) + "!!"
        min_value = float("inf")
        legalMoves = gameState.getLegalActions(agentIndex)
        for action in legalMoves:
            successorGameState = gameState.generateSuccessor(agentIndex, action)
            _, value = self.getValue(successorGameState, agentIndex, depth)
            #print "In min got value "+str(value)
            if value < min_value:
                min_value = value
                min_action = action

        #print "Min returning "+str(min_value)
        return (min_action, min_value)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

