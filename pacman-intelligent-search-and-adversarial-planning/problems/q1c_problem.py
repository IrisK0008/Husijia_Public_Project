import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1c_problem:
    """
    A search problem associated with finding a path that collects all of the
    food (dots) in a Pacman game.
    Some useful data has been included here for you
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState

    @log_function
    def getStartState(self):
        "*** YOUR CODE HERE ***"
        return (self.startingGameState.getPacmanPosition(), self.startingGameState.getFood())
    
    @log_function
    def isGoalState(self, state):
        "*** YOUR CODE HERE ***"
        pos, food = state
        return food.count() == 0

    @log_function
    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        "*** YOUR CODE HERE ***"
        successors = []
        pos, food = state
        x, y = pos

        # direction vectors 
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nx, ny = int(x + dx), int(y + dy)

            if not self.startingGameState.hasWall(nx, ny):
                # use graph search, take current state
                next_food = food.copy()
                # if there's food at the next position, eat it (mark it as False)
                if next_food[nx][ny]:
                    next_food[nx][ny] = False
                successors.append((((nx, ny), next_food), action, 1))
        return successors

