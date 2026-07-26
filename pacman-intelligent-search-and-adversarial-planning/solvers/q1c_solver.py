#---------------------#
# DO NOT MODIFY BEGIN #
#---------------------#

import logging

import util
from problems.q1c_problem import q1c_problem

#-------------------#
# DO NOT MODIFY END #
#-------------------#
def q1c_solver(problem):
    from game import Actions, Directions
    import util

    # =========================================================================
    # Phase One: Initialization and Static Environment Data Extraction
    # =========================================================================
    # initialize information
    start_pos = problem.startingGameState.getPacmanPosition()
    walls = problem.startingGameState.getWalls()
    original_food = problem.startingGameState.getFood().asList()
    food_list = list(original_food)
    initial_food_count = len(original_food)
    
    # record the full path of actions to take all food
    full_path = []
    current_pos = start_pos
    
    # if the starting position has food, eat it first
    if current_pos in food_list:
        food_list.remove(current_pos)
    last_action = Directions.STOP

    # =========================================================================
    # Phase Two: Heuristic Function Definition
    # =========================================================================
    #  ----- Custom Function: Calculate Manhattan Minimum Spanning Tree (MST) ----- 
    def get_manhattan_mst(unvisited_nodes):
        # If there is no food left, the baseline cost is 0
        if not unvisited_nodes: return 0

        nodes = list(unvisited_nodes)
        n = len(nodes)
        # If only 1 dot remains, no connections are needed, cost is 0 (walking cost is handled by len(path))
        if n == 1: return 0
        
        # Prim's Algorithm initialization
        mst_cost = 0
        min_dist = [float('inf')] * n
        min_dist[0] = 0
        in_mst = [False] * n
        
        # Prim's algorithm main loop: connect all n dots into the network
        for _ in range(n):
            best_d = float('inf')
            u = -1
            # Step 1: Find the closest unconnected dot to the current network
            for i in range(n):
                if not in_mst[i] and min_dist[i] < best_d:
                    best_d = min_dist[i]
                    u = i
            # If none found (e.g., unreachable dead ends), terminate early
            if u == -1: break
            # Step 2: Add the selected dot 'u' to the network
            in_mst[u] = True
            mst_cost += best_d
            # Step 3: Update distances of other unconnected dots from the newly added 'u'
            ux, uy = nodes[u]
            for v in range(n):
                if not in_mst[v]:
                    vx, vy = nodes[v]
                    # Calculate Manhattan distance from u to v
                    d = abs(ux - vx) + abs(uy - vy)
                    # If this new connection is shorter than v's previously recorded distance, update it
                    if d < min_dist[v]:
                        min_dist[v] = d
        # Finally, return the minimum guaranteed steps h(n) to connect all remaining dots
        return mst_cost
    

    # =========================================================================
    # Phase Three: Main Search Loop
    # =========================================================================
    while food_list:
        queue = util.Queue()  # Clear the old queue before searching for new dots
        
        # Pack the start position and an "empty action list" into the queue; [] means Pac-Man hasn't taken the first step in this round
        queue.push((current_pos, []))
        visited = {current_pos}  
        
        closest_foods = []
        min_dist = -1
        
        # ---------------------------------------------------------------------
        # Step 3.1: Candidate Target Node Generation based on Breadth-First Search (BFS)
        # ---------------------------------------------------------------------
        while not queue.isEmpty():
            pos, path = queue.pop()
            
            # Greedy strategy optimization for large state spaces: Early termination to prevent timeout
            if len(food_list) > 200:
                # As soon as the first wave of dots is hit, stop expanding deeper, cut it off!
                if min_dist != -1 and len(path) > min_dist:
                    break
                # Record the first wave of dots, stop recording subsequent waves
                if pos in food_list:
                    if min_dist == -1:
                        min_dist = len(path)
                    closest_foods.append((pos, path))
                    continue

            # Global exploration strategy for medium/small state spaces
            else:
                # After finding the first target, moderately expand search depth to collect more candidate nodes
                if min_dist != -1 and len(path) > min_dist + 2:
                    break  
                # Record food, find up to 20 dots to eat
                if pos in food_list:
                    if min_dist == -1:
                        min_dist = len(path)
                    closest_foods.append((pos, path))
                    # Set capacity limit for candidate pool, control search range to avoid performance issues (timeout or memory overflow)
                    if len(closest_foods) >= 20:
                        break

            # Inertia heuristic: Prioritize the movement direction from the last step, reducing meaningless turns
            curr_last_dir = path[-1] if path else last_action    
            directions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

            # Force the previous direction to the front of the check order for smooth straight lines
            if curr_last_dir in directions:
                directions.remove(curr_last_dir)
                directions.insert(0, curr_last_dir)
             
            x, y = pos
            # State expansion: Lightweight coordinate deduction based on static wall matrix
            for action in directions:
                dx, dy = Actions.directionToVector(action)
                nx, ny = int(x + dx), int(y + dy)
                npos = (nx, ny)

                # Boundary check and loop avoidance (prevent revisiting expanded nodes)
                if 0 <= nx < walls.width and 0 <= ny < walls.height:
                    if not walls[nx][ny] and npos not in visited:
                        visited.add(npos)  

                        queue.push((npos, path + [action]))
        
        # Exception handling: Terminate loop if no reachable food nodes are detected
        if not closest_foods:
            break

        best_food = None
        best_path = None
        
        # ---------------------------------------------------------------------
        # Step 3.2: Cost Evaluation and Optimal Selection of Candidate Target Nodes
        # ---------------------------------------------------------------------
        #  ----- Mode 1: Large Map Opening (> 200 dots) -- Prioritize isolated edge dots ----- 
        if len(food_list) > 200:
            max_isolation_dist = -1

            for food, path in closest_foods:
                # Send out a mini BFS probe (local_queue), starting from this candidate dot
                local_queue = util.Queue()
                local_queue.push((food, 0))
                local_visited = {food}
                isolation_dist = 0
                found = False

                # Local BFS to measure the shortest path from this candidate node to other remaining nodes
                while not local_queue.isEmpty():
                    local_pos, local_dist = local_queue.pop()
                    if local_pos in food_list and local_pos != food:
                        isolation_dist = local_dist
                        found = True
                        break    # Found the nearest neighbor, stop probing immediately
                    x2, y2 = local_pos
                    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                        dx, dy = Actions.directionToVector(action)
                        nx, ny = int(x2 + dx), int(y2 + dy)
                        npos = (nx, ny)
                        if 0 <= nx < walls.width and 0 <= ny < walls.height:
                            if not walls[nx][ny] and npos not in local_visited:
                                local_visited.add(npos)
                                local_queue.push((npos, local_dist + 1))
                if not found: isolation_dist = 0
                
                # Select the node with the highest isolation degree as the current optimal target
                if isolation_dist > max_isolation_dist or (isolation_dist == max_isolation_dist and (best_food is None or food < best_food)):
                    max_isolation_dist = isolation_dist
                    best_food = food
                    best_path = path
        

        #  ----- Mode 2: Medium/Small Map (<= 200 dots) -- "MST Global Cost Evaluation" ----- 
        else:
            # A* evaluation strategy: Minimize f(n) = g(n) + h(n)
            best_cost = float('inf')
            best_jump_dist = -1 
            for food, path in closest_foods:
                remaining = set(food_list) - {food}    

                # If this is the last dot, the cost is just the distance to walk there
                if not remaining:  
                    cost = len(path)
                    jump_dist = 0
                else:
                    # h(n) component 1: Shortest Manhattan distance from current node to the remaining cluster
                    jump_dist = min(abs(food[0] - r[0]) + abs(food[1] - r[1]) for r in remaining)

                    # h(n) component 2: Minimum Spanning Tree (MST) cost within the remaining cluster
                    mst_cost = get_manhattan_mst(remaining)

                    # Comprehensive evaluation function: Total cost f(n) = Path cost g(n) + Heuristic cost h(n)
                    cost = len(path) + jump_dist + mst_cost
                
                # Select the node with minimum total cost; tie-break by prioritizing nodes with longer jump distances
                if cost < best_cost or (cost == best_cost and jump_dist > best_jump_dist):
                    best_cost = cost
                    best_jump_dist = jump_dist
                    best_food = food
                    best_path = path
        

        # ---------------------------------------------------------------------
        # Step 3.3: Dynamic Pruning Optimization (Expected Return Evaluation in Endgame)
        # ---------------------------------------------------------------------
        #  ----- When less than 15 dots remain, enable "Accounting Mode". Stop eating if the time penalty exceeds the points gained -----
        if len(food_list) < 15:
            dist = len(best_path)
            # Calculate local density around the target node (number of neighboring nodes within distance <= 5)
            friends = sum(1 for f in food_list if f != best_food and (abs(best_food[0] - f[0]) + abs(best_food[1] - f[1])) <= 5)
            # Expected net value model = Base points + Scarcity premium - Distance time penalty + Local cluster reward
            expected_net_value = 10 + (500 / len(food_list)) - dist + 6 * friends
            # Pruning operation: If expected net value is negative, further exploration decreases total score, terminate search immediately
            if expected_net_value < 0:
                break
        
        # ---------------------------------------------------------------------
        # Step 3.4: Path Execution and State Synchronization Update
        # ---------------------------------------------------------------------
        curr_x, curr_y = current_pos[0], current_pos[1]
        for action in best_path:
            full_path.append(action)
            dx, dy = Actions.directionToVector(action)
            curr_x, curr_y = int(curr_x + dx), int(curr_y + dy)
            current_pos = (curr_x, curr_y)

            # Decouple planning and execution: While moving along the set path, dynamically clear other food nodes passed to avoid redundant target assignments in subsequent loops
            if current_pos in food_list:
                food_list.remove(current_pos)
        
        # Cache the last executed action for inertia heuristic in the next loop
        last_action = full_path[-1] if full_path else Directions.STOP
        
    return full_path