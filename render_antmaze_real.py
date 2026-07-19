import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import gymnasium_robotics


def get_maze_structure(env):
    """
    从 gymnasium_robotics 的 AntMaze 环境中提取真实的迷宫结构
    """
    unwrapped = env.unwrapped

    # gymnasium_robotics 的 AntMaze 环境通常有这些属性
    if hasattr(unwrapped, '_maze_map'):
        return unwrapped._maze_map

    if hasattr(unwrapped, 'maze_map'):
        return unwrapped.maze_map

    # 从 MuJoCo 模型中提取迷宫信息
    if hasattr(unwrapped, 'model'):
        return extract_maze_from_mujoco(unwrapped)

    return None


def extract_maze_from_mujoco(env):
    """
    从 MuJoCo 模型中提取迷宫结构
    """
    model = env.model
    data = env.data

    # 获取所有几何体的位置和大小
    walls = []
    free_spaces = []

    for i in range(model.ngeom):
        geom_name = model.geom(i).name if hasattr(model.geom(i), 'name') else ''
        geom_type = model.geom_type[i]
        geom_pos = model.geom_pos[i].copy()
        geom_size = model.geom_size[i].copy()

        # 检查是否是墙壁
        if 'wall' in geom_name.lower() or 'block' in geom_name.lower():
            walls.append({
                'pos': geom_pos[:2],  # x, y
                'size': geom_size[:2],
                'name': geom_name
            })

    return {'walls': walls}


def render_maze_topview(env_name='AntMaze_UMaze-v4', save_path='antmaze_real.png'):
    """
    渲染 AntMaze 环境的真实迷宫俯视图
    """
    print(f"Creating environment: {env_name}")
    env = gym.make(env_name)
    obs, info = env.reset()

    fig, ax = plt.subplots(figsize=(14, 14))

    # 尝试获取迷宫结构
    maze_info = get_maze_structure(env)

    if maze_info is not None and isinstance(maze_info, dict) and 'walls' in maze_info:
        # 从 MuJoCo 模型绘制
        walls = maze_info['walls']
        print(f"Found {len(walls)} walls from MuJoCo model")

        # 绘制背景（自由空间）
        if walls:
            all_x = [w['pos'][0] for w in walls]
            all_y = [w['pos'][1] for w in walls]
            all_size_x = [w['size'][0] for w in walls]
            all_size_y = [w['size'][1] for w in walls]

            min_x = min([x - s for x, s in zip(all_x, all_size_x)])
            max_x = max([x + s for x, s in zip(all_x, all_size_x)])
            min_y = min([y - s for y, s in zip(all_y, all_size_y)])
            max_y = max([y + s for y, s in zip(all_y, all_size_y)])

            # 绘制白色背景
            bg_rect = Rectangle((min_x, min_y), max_x-min_x, max_y-min_y,
                               facecolor='white', edgecolor='none', zorder=0)
            ax.add_patch(bg_rect)

            # 绘制墙壁
            for wall in walls:
                pos = wall['pos']
                size = wall['size']
                rect = Rectangle((pos[0]-size[0], pos[1]-size[1]),
                               size[0]*2, size[1]*2,
                               facecolor='#2C3E50', edgecolor='#34495E', linewidth=1, zorder=1)
                ax.add_patch(rect)

            # 获取并绘制起点和终点位置
            if 'achieved_goal' in obs and 'desired_goal' in obs:
                start_pos = obs['achieved_goal'][:2] if len(obs['achieved_goal']) >= 2 else None
                goal_pos = obs['desired_goal'][:2] if len(obs['desired_goal']) >= 2 else None

                if start_pos is not None:
                    circle = Circle(start_pos, 0.5, facecolor='lightgreen',
                                  edgecolor='green', linewidth=2, zorder=2)
                    ax.add_patch(circle)
                    ax.text(start_pos[0], start_pos[1], 'Start', ha='center', va='center',
                           fontsize=10, fontweight='bold', zorder=3)

                if goal_pos is not None:
                    circle = Circle(goal_pos, 0.5, facecolor='lightcoral',
                                  edgecolor='red', linewidth=2, zorder=2)
                    ax.add_patch(circle)
                    ax.text(goal_pos[0], goal_pos[1], 'Goal', ha='center', va='center',
                           fontsize=10, fontweight='bold', zorder=3)

            margin = 2
            ax.set_xlim(min_x-margin, max_x+margin)
            ax.set_ylim(min_y-margin, max_y+margin)

    elif maze_info is not None and isinstance(maze_info, np.ndarray):
        # 如果是数组格式的迷宫
        print(f"Found maze array with shape: {maze_info.shape}")
        height, width = maze_info.shape

        for i in range(height):
            for j in range(width):
                cell = maze_info[i, j]
                if cell == 1:  # 墙壁
                    rect = Rectangle((j, height-i-1), 1, 1,
                                   facecolor='#2C3E50', edgecolor='#34495E', linewidth=1)
                    ax.add_patch(rect)
                else:  # 空地
                    rect = Rectangle((j, height-i-1), 1, 1,
                                   facecolor='white', edgecolor='lightgray', linewidth=0.5)
                    ax.add_patch(rect)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
    else:
        print("Could not extract maze structure, using environment render")
        # 使用环境自带的渲染
        frame = env.render()
        if frame is not None:
            ax.imshow(frame)
            ax.axis('off')

    ax.set_aspect('equal')
    ax.set_title(f'{env_name} - Real Maze Structure', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved maze visualization to: {save_path}")
    plt.close()

    env.close()


def render_all_antmaze_envs():
    """
    渲染所有可用的 AntMaze 环境
    """
    env_names = [
        'AntMaze_UMaze-v4',
        'AntMaze_UMaze-v5',
        'AntMaze_Medium-v4',
        'AntMaze_Medium-v5',
        'AntMaze_Large-v4',
        'AntMaze_Large-v5',
    ]

    for env_name in env_names:
        try:
            print(f"\n{'='*70}")
            print(f"Rendering: {env_name}")
            print('='*70)
            # 从环境名提取大小
            size_name = env_name.split('_')[1].split('-')[0].lower()
            save_path = f"antmaze_{size_name}_v{env_name[-1]}.png"
            render_maze_topview(env_name, save_path)
        except Exception as e:
            print(f"✗ Error rendering {env_name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    # 渲染 Large 环境
    print("Rendering AntMaze_Large environment...")
    try:
        render_maze_topview('AntMaze_Large-v4', 'antmaze_large_real.png')
    except Exception as e:
        print(f"Error with v4, trying v5: {e}")
        try:
            render_maze_topview('AntMaze_Large-v5', 'antmaze_large_real.png')
        except Exception as e2:
            print(f"Error with v5: {e2}")

    # 如果想渲染所有环境，取消注释
    # render_all_antmaze_envs()
