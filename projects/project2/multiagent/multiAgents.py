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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return betterEvaluationFunction(successorGameState) - (2 if action == Directions.STOP else 0)

def scoreEvaluationFunction(currentGameState: GameState):
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

    def searchAction(self, gameState, mode):
        def visit(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            actions = state.getLegalActions(agent)
            if not actions:
                return self.evaluationFunction(state), None
            nextAgent = (agent + 1) % state.getNumAgents()
            nextDepth = depth + (nextAgent == 0)
            bestAction = actions[0]
            value = -float('inf') if agent == 0 else float('inf')
            if agent != 0 and mode == 'expectimax':
                total = 0
                for action in actions:
                    total += visit(state.generateSuccessor(agent, action),
                                   nextAgent, nextDepth, alpha, beta)[0]
                return total / len(actions), bestAction
            for action in actions:
                child = visit(state.generateSuccessor(agent, action),
                              nextAgent, nextDepth, alpha, beta)[0]
                if agent == 0:
                    if child > value:
                        value, bestAction = child, action
                    if mode == 'alphabeta':
                        if value > beta:
                            break
                        alpha = max(alpha, value)
                else:
                    if child < value:
                        value, bestAction = child, action
                    if mode == 'alphabeta':
                        if value < alpha:
                            break
                        beta = min(beta, value)
            return value, bestAction
        return visit(gameState, 0, 0, -float('inf'), float('inf'))[1]

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.searchAction(gameState, 'minimax')

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.searchAction(gameState, 'alphabeta')

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.searchAction(gameState, 'expectimax')

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Use maze distance for food, reward reachable scared ghosts, avoid danger.
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    position = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood().asList()
    from collections import deque
    walls = currentGameState.getWalls()
    distances = {position: 0}
    queue = deque([position])
    while queue:
        x, y = queue.popleft()
        for neighbor in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
            nx, ny = neighbor
            if not walls[nx][ny] and neighbor not in distances:
                distances[neighbor] = distances[x, y] + 1
                queue.append(neighbor)
    score = currentGameState.getScore() - 4 * len(foods)
    if foods:
        score += 10.0 / (1 + min(distances.get(f, 999) for f in foods))
    score -= 12 * len(currentGameState.getCapsules())
    for ghost in currentGameState.getGhostStates():
        distance = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > distance:
            score += 200.0 / (1 + distance)
        else:
            if distance <= 1:
                score -= 500
            score -= 5.0 / (1 + distance)
    return score

# Abbreviation
better = betterEvaluationFunction
