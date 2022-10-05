import numpy as np

def generate_e_maze(height, width, deadends, hallway_width):
    maze = np.ones((height, width))

    # add the start and goal
    start_width = width // 2 - 1

    
    maze[-hallway_width:, :start_width + 1] = 0
    maze[-hallway_width:, start_width + 2:] = 0
    maze[:, :hallway_width] = 0
    maze[:, -hallway_width:] = 0
    maze[:hallway_width, :] = 0

    maze[-1, start_width] = 2
    maze[-1, start_width+2] = 3

    # from left to right, we define the first column is an empty hallway
    # second column must be a wall
    # deadends must produce a hallway + a right wall

    if deadends > 0:
        reserved_space = 2 * hallway_width + 1  # hallways on each side + second_column=wall
        space_per_deadend = (width - reserved_space) // deadends
        assert space_per_deadend >= hallway_width + 1, 'Invalid configuration requested'
        free_space = space_per_deadend - hallway_width
        left_free_space = free_space // 2
        right_free_space = free_space - left_free_space
        for deadend_idx in range(deadends):
            left_idx = hallway_width + 1 + deadend_idx * space_per_deadend + left_free_space
            right_idx = left_idx + hallway_width
            maze[1:-hallway_width-1, left_idx:right_idx] = 0
            

    # wrap maze
    wrapped_maze = np.ones((height+2, width+2))
    wrapped_maze[1:-1, 1:-1] = maze
    return wrapped_maze


generate_e_maze(5, 10, 3, 1)
