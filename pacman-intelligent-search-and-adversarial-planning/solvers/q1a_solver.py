#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1a_problem import q1a_problem

def q1a_solver(problem: q1a_problem):
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
        self.goal = None

def astar_initialise(problem: q1a_problem):
    # YOUR CODE HERE
    astarData = AStarData()
    start_state = problem.getStartState()
    astarData.goal = problem.startingGameState.getFood().asList()[0]
    start_node = (start_state, [], 0)
    
    # A* priority: f(n) = g(n) + h(n)
    priority = 0 + astar_heuristic(start_state, astarData.goal)
    
    astarData.pq.push(start_node, priority)

    return astarData

def astar_loop_body(problem: q1a_problem, astarData: AStarData):
    # YOUR CODE HERE

    # 1. If the priority queue is empty, the whole map was searched without finding the goal
    if astarData.pq.isEmpty():
        return True, []  
    
    current_state, actions, current_cost = astarData.pq.pop()

    # 2. If the current state is our goal, we are done
    if problem.isGoalState(current_state):
        return True, actions  

    # 3. If this state hasn't been truly visited (explored) yet
    if current_state not in astarData.visited:
        astarData.visited.add(current_state)

        for next_state, action, step_cost in problem.getSuccessors(current_state):
            if next_state not in astarData.visited:
                new_cost = current_cost + step_cost
                new_actions = actions + [action]
                priority = new_cost + astar_heuristic(next_state, astarData.goal)
                
                astarData.pq.push((next_state, new_actions, new_cost), priority)

    return False, []

def astar_heuristic(current, goal):
    # YOUR CODE HERE
    return util.manhattanDistance(current, goal)