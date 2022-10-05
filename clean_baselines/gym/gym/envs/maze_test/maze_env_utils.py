import numpy as np


def string_to_npy_maze(strings, maze):
    arrs = list()
    for row_idx, string in enumerate(strings):
        string = string.replace('+', '1')
        string = string.replace('-', '1')
        string = string.replace('|', '1')
        string = string.replace(' ', '0')
        string = string.replace('S', '2')
        string = string.replace('G', '3')
        arr = np.zeros(len(string))
        for idx, char in enumerate(string):
            arr[idx] = int(char)
        arrs.append(arr)
        # if not np.all(maze[row_idx] == arr):
        #     import pdb; pdb.set_trace()
        #     pass
    maze = np.array(arrs)
    return maze


def npy_to_string_maze(maze):
    n_rows, n_cols = maze.shape
    strings = []
    for row_idx in range(n_rows):
        # import pdb; pdb.set_trace()
        string = ''
        for col_idx in range(n_cols):
            current = maze[row_idx, col_idx]
            if current == 2:
                char = 'S'
            if current == 0:
                char = ' '
            elif current == 3:
                char = 'G'
            elif current == 1:
                # corners
                if (row_idx == 0 or row_idx == n_rows-1) and (col_idx == 0 or col_idx == n_cols-1):
                    char = '+'
                # top/bottom walls
                elif row_idx == 0 or row_idx == n_rows-1:
                    if row_idx == 0 and maze[1, col_idx] == 1:
                        char = '+'
                    elif row_idx == n_rows-1 and maze[-2, col_idx] == 1:
                        char = '+'
                    else:
                        char = '-'
                    
                # side walls
                elif col_idx == 0 or col_idx == n_cols-1:
                    if col_idx == 0 and maze[row_idx, 1] == 1:
                        char = '+'
                    elif col_idx == n_cols-1 and maze[row_idx, col_idx - 1] == 1:
                        char = '+'
                    else:
                        char = '|'
                else:
                    vertical = maze[row_idx-1, col_idx] == 1 or maze[row_idx+1, col_idx] == 1
                    horizontal = maze[row_idx, col_idx+1] == 1 or maze[row_idx, col_idx-1] == 1
                    if horizontal and vertical:
                        char = '+'
                    elif horizontal:
                        char = '-'
                    elif vertical:
                        char = '|'
                    else:
                        char = '-'
            string += char
        strings.append(string)
    return strings
