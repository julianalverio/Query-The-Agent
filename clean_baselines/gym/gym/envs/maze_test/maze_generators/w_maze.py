import numpy as np


def generate_w_maze(num_u, length_lower, length_upper, bridge_length_lower, bridge_length_upper):
    bridge_lengths = np.random.randint(low=bridge_length_lower, high=bridge_length_upper+1, size=num_u+1)
    w_lengths = np.random.randint(low=length_lower, high=length_upper+1, size=num_u+1)
    max_width, min_width, value = 0, 0, 0
    for idx, w_length in enumerate(w_lengths):
        if idx % 2 == 0:
            value += w_length
        else:
            value -= w_length
        if value > max_width:
            max_width = value
        elif value < min_width:
            min_width = value

    maze_width = abs(min_width) + max_width
    maze_height = np.sum(bridge_lengths[:-1]) + num_u + 1
    
    # maze_height = num_u*bridge_length + num_u+1
    
    maze = np.ones((maze_height, maze_width))
    current = maze_width - max_width
    current_height = 0
    starting_position = current
    for u_idx, (w_length, bridge_length) in enumerate(zip(w_lengths, bridge_lengths)):
        if u_idx % 2 == 0: # going right
            maze[current_height, current:current+w_length] = 0  # hallway
            if u_idx != num_u:
                maze[current_height+1:current_height+1+bridge_length, current+w_length-1] = 0  # bridge
        else:  # going left
            maze[current_height, current - w_length + 1:current+1] = 0  # hallway
            if u_idx != num_u:
                maze[current_height+1:current_height+1+bridge_length, current-w_length+1] = 0  # bridge

        current_height += bridge_length + 1
        if u_idx % 2 == 0:
            current += w_length - 1
        else:
            current -= w_length - 1
        # print(w_length)
        # print(f'current {current}, current height {current_height}')
        # print(maze)
    maze[0, starting_position] = 2 # start
    maze[-1, current] = 3  # goal
    wrapped_maze = np.ones((maze.shape[0]+2, maze.shape[1]+2))
    wrapped_maze[1:-1, 1:-1] = maze
    return wrapped_maze.T  # transpose so the maze moves left to right


if __name__ == '__main__':
    print(generate_w_maze(num_u=6, length_lower=1, length_upper=7, bridge_length_lower=1, bridge_length_upper=1))
    print(generate_w_maze(num_u=6, length_lower=3, length_upper=3, bridge_length_lower=1, bridge_length_upper=1))

