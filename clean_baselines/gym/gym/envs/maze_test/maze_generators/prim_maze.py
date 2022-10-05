import numpy as np

def get_neighbors_and_visit_count(location, maze, height, width):
    y, x = location
    neighbors = list()
    if y >= 1:
        neighbors.append((y-1, x))
    if y <= height - 2:
        neighbors.append((y+1, x))
    if x >= 1:
        neighbors.append((y, x-1))
    if x <= width - 2:
        neighbors.append((y, x+1))

    visit_count = 0
    for neighbor in neighbors:
        if maze[neighbor[0], neighbor[1]] == 0:
            visit_count += 1
    return neighbors, visit_count

def generate_prim_maze(height, width):
    inner_maze = np.ones((height, width))
    start_y = np.random.randint(height)
    start_x = np.random.randint(width)
    inner_maze[start_y, start_x] = 0
    neighbors, _ = get_neighbors_and_visit_count((start_y, start_x), inner_maze, height, width)
    queue = neighbors
    all_reachable = [(start_y, start_x)]
    all_reachable.extend(neighbors)
    while queue:
        pop_idx = np.random.randint(len(queue))
        location = queue.pop(pop_idx)
        neighbors, visit_count = get_neighbors_and_visit_count(location, inner_maze, height, width)
        all_reachable.extend(neighbors)
        if visit_count != 1:
            continue
        inner_maze[location[0], location[1]] = 0
        queue.extend(neighbors)

    end_x = np.random.choice(np.where(inner_maze[0]==0)[0])
    start_x = np.random.choice(np.where(inner_maze[-1]==0)[0])

    # now add the outer perimeter
    maze = np.ones((height+2, width+2))
    maze[1:-1, 1:-1] = inner_maze

    # set the start and goal
    maze[1, end_x] = 2
    maze[-2, start_x] = 3

    return maze


if __name__ == '__main__':
    maze = generate_prim_maze(10, 10)
    print(maze)
    
