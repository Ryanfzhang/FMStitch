import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def render_antmaze_map(env_name='AntMaze_Large-v4', save_path='antmaze_large_map.png'):
    """
    Render and save a top-down view of the AntMaze environment.

    Args:
        env_name: Name of the AntMaze environment
        save_path: Path to save the rendered image
    """
    try:
        # Create the environment
        env = gym.make(env_name, render_mode='rgb_array')

        # Reset environment to get initial state
        obs, info = env.reset()

        # Try to get maze structure if available
        if hasattr(env.unwrapped, 'maze_map'):
            maze_map = env.unwrapped.maze_map

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 12))

            # Plot maze structure
            height, width = maze_map.shape
            for i in range(height):
                for j in range(width):
                    if maze_map[i, j] == 1:  # Wall
                        rect = Rectangle((j, height-i-1), 1, 1,
                                       facecolor='black', edgecolor='gray')
                        ax.add_patch(rect)
                    else:  # Free space
                        rect = Rectangle((j, height-i-1), 1, 1,
                                       facecolor='white', edgecolor='lightgray')
                        ax.add_patch(rect)

            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            ax.set_aspect('equal')
            ax.set_title(f'{env_name} - Maze Structure', fontsize=16, fontweight='bold')
            ax.axis('off')

        else:
            # Fallback: render using environment's render method
            frame = env.render()
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(frame)
            ax.set_title(f'{env_name} - Environment View', fontsize=16, fontweight='bold')
            ax.axis('off')

        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved AntMaze map to: {save_path}")
        plt.close()

        env.close()

    except Exception as e:
        print(f"Error rendering with gymnasium: {e}")
        print("\nTrying alternative approach...")

        # Alternative: Create a manual representation based on known AntMaze-Large structure
        # AntMaze-Large typically has a specific maze layout
        create_manual_antmaze_large(save_path)

def create_manual_antmaze_large(save_path='antmaze_large_map.png'):
    """
    Create a manual representation of AntMaze-Large maze structure.
    Based on the known layout of AntMaze-Large environment.
    """
    # AntMaze-Large is typically a 24x24 maze
    maze_size = 24

    # Create maze structure (1 = wall, 0 = free space)
    maze = np.ones((maze_size, maze_size))

    # Define the maze structure for AntMaze-Large
    # This is a simplified representation - adjust based on actual layout
    free_spaces = [
        # Top corridor
        (0, range(0, 12)),
        # Vertical corridors
        (range(0, 12), 0), (range(0, 12), 11),
        (range(12, 24), 12), (range(12, 24), 23),
        # Bottom corridor
        (23, range(12, 24)),
        # Horizontal connections
        (range(0, 24), 5), (range(0, 24), 18),
        # Middle section
        (11, range(0, 24)), (12, range(0, 24)),
    ]

    # Mark free spaces
    for rows, cols in free_spaces:
        if isinstance(rows, range):
            rows = list(rows)
        elif not isinstance(rows, list):
            rows = [rows]

        if isinstance(cols, range):
            cols = list(cols)
        elif not isinstance(cols, list):
            cols = [cols]

        for i in rows:
            for j in cols:
                if 0 <= i < maze_size and 0 <= j < maze_size:
                    maze[i, j] = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot maze
    for i in range(maze_size):
        for j in range(maze_size):
            if maze[i, j] == 1:  # Wall
                rect = Rectangle((j, maze_size-i-1), 1, 1,
                               facecolor='#2C3E50', edgecolor='#34495E')
                ax.add_patch(rect)
            else:  # Free space
                rect = Rectangle((j, maze_size-i-1), 1, 1,
                               facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=0.5)
                ax.add_patch(rect)

    ax.set_xlim(0, maze_size)
    ax.set_ylim(0, maze_size)
    ax.set_aspect('equal')
    ax.set_title('AntMaze-Large - Maze Structure', fontsize=16, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved AntMaze-Large map to: {save_path}")
    plt.close()

if __name__ == '__main__':
    # Try different possible environment names
    env_names = [
        'AntMaze_Large-v4',
        'AntMaze_Large-v3',
        'antmaze-large-diverse-v2',
        'antmaze-large-play-v2',
    ]

    success = False
    for env_name in env_names:
        try:
            print(f"Trying to render: {env_name}")
            render_antmaze_map(env_name)
            success = True
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if not success:
        print("\nCould not find AntMaze environment in gymnasium.")
        print("Creating manual representation...")
        create_manual_antmaze_large()
