import gym
import d4rl  # 必须导入以注册环境
import numpy as np
from PIL import Image
import os
os.environ['MUJOCO_GL'] = 'osmesa'

def render_antmaze_large(save_path='antmaze_large_render.png'):
    """
    使用 D4RL 渲染 AntMaze-large 的图像
    需要安装: pip install gymnasium d4rl mujoco
    """
    # 创建环境
    env = gym.make('antmaze-large-play-v2', render_mode='rgb_array')
    
    # 重置环境到随机状态
    obs = env.reset()
    
    # 渲染一帧
    frame = env.render()
    
    # 转换为 PIL Image 并保存
    img = Image.fromarray(frame)
    img.save(save_path)
    print(f"Rendered image saved to {save_path}")
    
    # 也可以保存多个视角
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, ax in enumerate(axes):
        obs, info = env.reset()
        # 让蚂蚁移动几步以获得不同视角
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        ax.imshow(frame)
        ax.set_title(f'View {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('antmaze_large_multi_view.png', dpi=300, bbox_inches='tight')
    env.close()

if __name__ == '__main__':
    render_antmaze_large()
