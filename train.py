import os
import torch
import numpy as np
import pybullet as p

from envs.ArmPickAndDrop import ArmPickAndDrop
from envs.robot import UR5Robotiq85
from helper.utilities import YCBModels, Camera
from rl_algorithm.TD3 import TD3, ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def td3_trainning():
    robot = UR5Robotiq85((0, 0.5, 0), (0, 0, 0))
    camera = Camera((1, 1, 1), (0, 0, 0), (0, 0, 1), 0.1, 5, (320, 320), 40)
    target_position_B = np.array([0.5, 0.5, 0.0])
    ycb_models = YCBModels(
        os.path.join("./data/ycb", "**", "textured-decmp.obj"),
    )
    env = ArmPickAndDrop(
        robot, ycb_models, camera, vis=False, target_position_B=target_position_B
    )

    state = env.reset()
    state_dim = len(state)
    action_dim = 7  # robot.get_action_space()
    max_action = 1.0
    td3 = TD3(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    num_episodes = 1000
    replay_buffer = ReplayBuffer()
    batch_size = 256

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = td3.select_action(
                torch.tensor(state, dtype=torch.float32, device=device)
            )

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(
                torch.tensor(state, dtype=torch.float32, device=device),
                torch.tensor(action, dtype=torch.float32, device=device),
                torch.tensor(next_state, dtype=torch.float32, device=device),
                torch.tensor(reward, dtype=torch.float32, device=device),
                torch.tensor(done, dtype=torch.float32, device=device),
            )

            if len(replay_buffer) > batch_size:
                td3.train(replay_buffer, batch_size)

            state = next_state
            episode_reward += reward

        print(f"Episode {episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    td3_trainning()
