import numpy as np

def generate_block_maze(height, width, free_blocks=1):
    inner_maze = np.zeros((height, width))
    inner_maze[free_blocks, free_blocks:width - free_blocks] = 1  # top
    inner_maze[height - free_blocks - 1, free_blocks:width - free_blocks] = 1  # bottom
    inner_maze[free_blocks:height - free_blocks, free_blocks] = 1  # left
    inner_maze[free_blocks:height - free_blocks, height - free_blocks - 1] = 1  # right

    # start and goal. Start at the bottom and move to the top
    inner_maze[-1, width//2] = 2
    inner_maze[0, width//2] = 3

    
    # wrap it
    maze = np.ones((height+2, width+2))
    maze[1:-1, 1:-1] = inner_maze
    return maze

if __name__ == '__main__':
    print(generate_block_maze(10, 10, 4))
    
    
