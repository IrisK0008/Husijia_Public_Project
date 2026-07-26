#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1b_problem import q1b_problem

def q1b_solver(problem: q1b_problem):
    astarData = astar_initialise(problem)
    num_expansions = 0
    terminate = False
    while not terminate:
        num_expansions += 1
        terminate, result = astar_loop_body(problem, astarData)
    print(f'Number of node expansions: {num_expansions}')
    return result

#-------------------#
# DO NOT MODIFY END #
#-------------------#

class AStarData:
    def __init__(self):
        self.pq = util.PriorityQueue()
        self.visited = set() 

def astar_initialise(problem: q1b_problem):
    # YOUR CODE HERE
    astarData = AStarData()
    start_state = problem.getStartState()
    h_cost = astar_heuristic(start_state, problem)

    # Push start state: (state, path, g_cost), priority: (f_cost, h_cost)
    astarData.pq.push((start_state, [], 0), (h_cost, h_cost))
    return astarData

def astar_loop_body(problem: q1b_problem, astarData: AStarData):
    # YOUR CODE HERE
    current_state = None
    path = None
    g_cost = 0

    # Extract the highest priority (lowest f_cost) unvisited node
    while not astarData.pq.isEmpty():
        current_state, path, g_cost = astarData.pq.pop()
        if current_state not in astarData.visited:
            break 
    else:
        return True, []

    astarData.visited.add(current_state)

    if problem.isGoalState(current_state):
        return True, path
    
    # Expand next node and evaluate valid successors
    for next_state, action, step_cost in problem.getSuccessors(current_state):
        # Only consider unvisited states to avoid cycles
        if next_state not in astarData.visited:
            new_cost = g_cost + step_cost
            h_cost = astar_heuristic(next_state, problem)
            f_cost = new_cost + h_cost
            priority = (f_cost, h_cost)
            new_path = path + [action]

            astarData.pq.push((next_state, new_path, new_cost), priority)

    return False, []  # Not terminating yet,expand next node

def astar_heuristic(state, problem):
    # YOUR CODE HERE
    # Multi-source Breadth-First Search (BFS) to precompute shortest paths from all food pellets
    # Check if we need to initialize or update the cached distance data for the current map
    if not hasattr(astar_heuristic, 'radar_data') or getattr(astar_heuristic, 'current_map', None) != problem.startingGameState:
        astar_heuristic.current_map = problem.startingGameState
        astar_heuristic.radar_data = {}
        
        walls = problem.startingGameState.getWalls()
        food_list = problem.startingGameState.getFood().asList()
        
        # Treat all food pellets as multiple starting points (distance = 0)
        queue = []
        for food in food_list:
            queue.append((food, 0))
            astar_heuristic.radar_data[food] = 0
            
        # Propagate outwards from all food locations simultaneously 
        head = 0
        while head < len(queue):
            (cx, cy), dist = queue[head]
            head += 1
            # Explore the 4 adjacent directions (up, down, right, left)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                # Check if the new coordinate is within the map boundaries
                if 0 <= nx < walls.width and 0 <= ny < walls.height: 
                    if not walls[nx][ny]:
                        # If the cell hasn't been reached yet, record its distance and add to queue.
                        # Since it's BFS, the first time we reach a cell guarantees the absolute shortest distance.
                        if (nx, ny) not in astar_heuristic.radar_data:
                            astar_heuristic.radar_data[(nx, ny)] = dist + 1
                            queue.append(((nx, ny), dist + 1))
    
    # Look up the precomputed distance for the current state in O(1) time. 
    # Return infinity if the state is completely isolated/unreachable.
    return astar_heuristic.radar_data.get(state, float('inf'))