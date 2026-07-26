import logging
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class Q2_Agent(Agent):

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '3'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
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
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        "*** YOUR CODE HERE ***"

        # =====================================================================
        # Part 1: Core Alpha-Beta Pruning (Adversarial Search)
        # =====================================================================
        def alpha_beta(state, agentIndex, depth, alpha, beta):
            # 1. Termination conditions: Reached max depth or game over (win/loss)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            legalActions = state.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(state)
            
            # Dynamic Ghost Pruning
            # If it's a ghost's turn, calculate its Manhattan distance to Pacman. 
            # If distance > 5, it poses no immediate threat within the search horizon. 
            # restrict its actions to just 1 (skipping branching) to drastically 
            if agentIndex > 0: 
                ghost_pos = state.getGhostPosition(agentIndex)
                pacman_pos = state.getPacmanPosition()
                if abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) > 5:
                    legalActions = [legalActions[0]]

            # 2. Max Node (Pacman's turn)
            if agentIndex == 0:  
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alpha_beta(successor, 1, depth, alpha, beta))
                    # Alpha-Beta Pruning: If value exceeds beta, prune remaining branches
                    if value > beta:    
                        return value
                    alpha = max(alpha, value)
                return value
            
            # 3. Min Node (Ghost's turn)
            else:  
                value = float('inf')
                nextAgent = agentIndex + 1

                # Turn rotation: If all ghosts have moved, increment depth and pass turn back to Pacman
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                else:
                    nextDepth = depth
                
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(value, alpha_beta(successor, nextAgent, nextDepth, alpha, beta))
                    # Alpha-Beta Pruning: If value is less than alpha, prune remaining branches
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        
        # =====================================================================
        # Part 2: Root Node Action Evaluation & Optimal Solution Extraction
        # =====================================================================
        legalActions = gameState.getLegalActions(0)
        
        # Fallback: If trapped with no valid actions, return STOP to prevent crashes
        if not legalActions:
            return Directions.STOP
        # Force movement: Remove STOP to avoid time penalty (-1 point per turn) unless trapped
        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)
        
        best_actions = []
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        successors_cache = {}

        # Action Heuristic Ordering (Move Ordering)
        # Pre-evaluate and sort actions to maximize Alpha-Beta pruning efficiency
        action_candidates = []
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            successors_cache[action] = successor
            rough_score = self.evaluationFunction(successor)
            action_candidates.append((action, successor, rough_score))
            
        action_candidates.sort(key=lambda x: x[2], reverse=True)

        for action, successor, _ in action_candidates:
            score = alpha_beta(successor, 1, 0, alpha, beta)
            
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
            
            alpha = max(alpha, best_score)
        
        # =====================================================================
        # Part 3: Tie-breaking Strategy - Environment-Aware BFS
        # =====================================================================
        # If Alpha-Beta finds multiple equally safe actions, trigger BFS for optimal pathfinding
        if len(best_actions) > 1:
            walls = gameState.getWalls()
            pacman_pos = gameState.getPacmanPosition()
            ghosts = gameState.getGhostStates()
            
            scared_ghost_positions = set()
            danger_zone = set() 
            normal_ghost_dists = []
            
            # Danger Zone Mapping
            for g in ghosts:
                gx, gy = int(g.getPosition()[0]), int(g.getPosition()[1])
                if g.scaredTimer > 0:
                    scared_ghost_positions.add((gx, gy))     # Record vulnerable scared ghosts
                else:
                    normal_ghost_dists.append(abs(pacman_pos[0] - gx) + abs(pacman_pos[1] - gy))
                    # Mark normal ghosts and their 4 adjacent tiles as an "Absolute Forbidden Zone"
                    danger_zone.update([(gx, gy), (gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)])

            min_ghost_dist = min(normal_ghost_dists) if normal_ghost_dists else float('inf')

            # Dynamic target switching: If ghosts are closing in (dist <= 5), prioritize 
            # capsules for survival; otherwise, prioritize hunting scared ghosts.
            hunt_targets = set()
            if min_ghost_dist <= 5:
                food_set = set(gameState.getFood().asList())
                capsule_set = set(gameState.getCapsules())
                hunt_targets.update(capsule_set)
            hunt_targets.update(scared_ghost_positions)
            
            hunt_mode = len(hunt_targets) > 0

            # Single-Wave Fast BFS 
            # Time complexity optimization: Casts concurrent waves from safe actions and returns
            # immediately upon hitting the first target, ensuring optimal pathing with low overhead.
            def single_wave_bfs(start_pos, target_set):
                if not target_set: return None
                if start_pos in target_set: return Directions.STOP
                
                queue = util.Queue()
                visited = {start_pos}
                
                # Initial expansion: Only cast waves along Alpha-Beta verified safe routes
                for action in best_actions:
                    nx, ny = successors_cache[action].getPacmanPosition()
                    pos = (nx, ny)
                    
                    if pos in danger_zone: continue
                    
                    if pos not in visited:
                        if pos in target_set: return action
                        visited.add(pos)
                        queue.push((pos, 1, action))    # Carry the root action in the queue
                        
                while not queue.isEmpty():
                    curr, d, root_action = queue.pop()
                    x, y = curr
                    
                    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                        nx, ny = int(x + dx), int(y + dy)
                        pos = (nx, ny)
                        
                        if not walls[nx][ny] and pos not in visited:
                            if pos in danger_zone: continue   # Avoid ghost perimeters
                            
                            # Upon touching any target, instantly return the root action of this wave
                            if pos in target_set: 
                                return root_action
                            visited.add(pos)
                            queue.push((pos, d + 1, root_action))
                                
                return None

            best_action = None
            
            # Prioritize Hunt/Escape mode (finding capsules or eating scared ghosts)
            if hunt_mode:
                best_action = single_wave_bfs(pacman_pos, hunt_targets)
            
            # If not in hunt mode, or the hunt path is blocked, search for normal food dots
            if not hunt_mode or best_action is None:
                food_set = set(gameState.getFood().asList())
                capsule_set = set(gameState.getCapsules())
                best_action = single_wave_bfs(pacman_pos, food_set)

            if best_action is not None:
                return best_action

        return best_actions[0]