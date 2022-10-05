import numpy as np

def generate_empty_maze():
    maze = np.zeros((5, 5))
    maze[0, 0] = 2
    maze[-1, -1] = 3
    return maze
