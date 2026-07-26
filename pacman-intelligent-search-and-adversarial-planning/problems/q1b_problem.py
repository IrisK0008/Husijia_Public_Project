import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    def __init__(self, gameState: GameState):
        self.startingGameState: GameState = gameState

    @log_function
    def getStartState(self):
        return self.startingGameState.getPacmanPosition()

    @log_function
    def isGoalState(self, state):
        x, y = state
        return self.startingGameState.hasFood(x, y)

    @log_function
    def getSuccessors(self, state):
        successors = []
        x, y = state
        # # Iterate through the four possible moving directions
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            next_x, next_y = int(x + dx), int(y + dy)
            # Check if the next coordinate is a wall
            if not self.startingGameState.hasWall(next_x, next_y):
                next_state = (next_x, next_y)
                successors.append((next_state, action, 1))
        return successors
