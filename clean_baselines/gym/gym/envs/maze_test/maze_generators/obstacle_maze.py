# from envs import EnvWithGoal
import numpy as np


def generate_obstacle_maze(height, width, density=0.3):
    height += 2
    width += 1

    success = False
    attempts = 0
    while not success:
        attempts += 1
        inner_maze = (np.random.uniform(size=(height, width)) < density).astype(np.uint8)
        # the maze adds an empty row on the bottom to start, and an empty row at the top as a goal region,
        # and a perimeter on the sides and bottom
        maze = np.zeros((height+4, width+2))
        maze[2:-2, 1:-1] = inner_maze
        maze[[0, -1], :] = 1  # top and bottom walls
        maze[:, [0, -1]] = 1  # side walls
        maze[1, np.random.randint(width)] = 3  # goal
        maze[-2, np.random.randint(width)] = 2
        success = solve_maze(maze)
    print(f'Generated maze in {attempts} attempt(s)')
    return maze

def get_children(location, height, width, seen, maze):
    y, x = location
    children = list()
    if x <= width - 2:
        children.append((y, x+1))
    if x >= 1:
        children.append((y, x-1))
    if y <= height - 2:
        children.append((y+1, x))
    if y >= 1:
        children.append((y-1, x))
    verified_children = list()
    for child in children:
        if child in seen:
            continue
        if maze[child[0], child[1]] == 1:
            continue
        seen.add(child)
        verified_children.append(child)
    return verified_children

def check_done(location):
    return location[0] == 3

# Depth-first search implementation to check if maze is solvable
def solve_maze(maze):
    height, width = maze.shape
    seen = set()
    queue = list()
    for x_coord in range(width):
        y_coord = height - 2
        if maze[y_coord, x_coord] == 1:
            continue
        queue.append((y_coord, x_coord))
        seen.add((y_coord, x_coord))
    while queue:
        location = queue.pop()
        if check_done(location):
            return True
        children = get_children(location, height, width, seen, maze)
        queue.extend(children)
    return False


if __name__ == '__main__':
    maze = generate_obstacle_maze(10, 10, 0.3)
    print(maze)
    
