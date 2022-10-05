from gym.envs.maze_test.maze_generators.obstacle_maze import generate_obstacle_maze  # height, width, optional_density
from gym.envs.maze_test.maze_generators.prim_maze import generate_prim_maze  # height, width
from gym.envs.maze_test.maze_generators.w_maze import generate_w_maze  # num_uy, length, bridge_length
from gym.envs.maze_test.maze_generators.block_maze import generate_block_maze
from gym.envs.maze_test.maze_manager_2d import MazeEnv2D
from gym.envs.maze_test.maze_generators.e_maze import generate_e_maze  # height, width, deadends, hallway_width
from gym.envs.maze_test.maze_generators.empty_maze import generate_empty_maze
from gym.envs.maze_test.maze_manager_3d import MazeEnv3D


def generate_maze(args):
    grid = generate_maze_grid(args)

    maze_2d = MazeEnv2D(grid, args)
    if args.maze_dimensionality == 3:
        env = MazeEnv3D(maze_2d, args)
        return env
    else:
        return maze_2d

def generate_maze_grid(args):
    assert args.maze_dimensionality in (2, 3)
    assert args.maze_type in ('w', 'prim', 'obstacle', 'e', 'empty', 'block')
    args.maze_type = args.maze_type.lower()

    if args.maze_type == 'w':
        maze = generate_w_maze(args.maze_num_u, args.maze_w_length_lower, args.maze_w_length_upper, args.maze_bridge_length_lower, args.maze_bridge_length_upper)
    elif args.maze_type == 'prim':
        maze = generate_prim_maze(args.maze_height, args.maze_width)
    elif args.maze_type == 'obstacle':
        maze = generate_obstacle_maze(args.maze_height, args.maze_width, args.obstacle_density)
    elif args.maze_type == 'block':
        maze = generate_block_maze(args.maze_height, args.maze_width, args.maze_free_blocks)
    elif args.maze_type == 'e':
        maze = generate_e_maze(args.maze_height, args.maze_width, args.maze_num_deadends, args.maze_hallway_width)
    elif args.maze_type == 'empty':
        maze = generate_empty_maze()
    return maze
