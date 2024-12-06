import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from utilities import Models, Camera, logger
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class ClutteredPushGrasp:

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(self, robot, models: Models, camera=None, vis=False):
        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera

        # define environment
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")

        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.xin = p.addUserDebugParameter("x", -0.224, 0.224, 0)
        self.yin = p.addUserDebugParameter("y", -0.224, 0.224, 0)
        self.zin = p.addUserDebugParameter("z", 0, 1.0, 0.5)
        self.rollId = p.addUserDebugParameter("roll", -3.14, 3.14, 0)
        self.pitchId = p.addUserDebugParameter("pitch", -3.14, 3.14, np.pi / 2)
        self.yawId = p.addUserDebugParameter("yaw", -np.pi / 2, np.pi / 2, np.pi / 2)
        self.gripper_opening_length_control = p.addUserDebugParameter(
            "gripper_opening_length", 0, 0.085, 0.04
        )

        self.boxID = p.loadURDF(
            "assets/urdf/block.urdf",
            [0.0, 0.0, 0.0],
            # p.getQuaternionFromEuler([0, 1.5706453, 0]),
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION,
        )

        # For calculating the reward
        self.box_opened = False
        self.btn_pressed = False
        self.box_closed = False

    def step_simulation(self):
        """
        Run one simulation step in PyBullet.

        This function hooks into the PyBullet `stepSimulation` method to advance the simulation by one step.
        If visualization is enabled, it adds a delay and updates the progress bar.

        Parameters:
        - None

        Returns:
        - None
        """
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)

    def read_debug_parameter(self):
        """
        Read user-defined debug parameters from PyBullet's debug sliders.

        Parameters:
        - None

        Returns:
        - Tuple: (x, y, z, roll, pitch, yaw, gripper_opening_length), where:
            - x, y, z: Cartesian coordinates of the robot end effector.
            - roll, pitch, yaw: Orientation angles of the end effector in radians.
            - gripper_opening_length: The opening length of the gripper.
        """
        x = p.readUserDebugParameter(self.xin)
        y = p.readUserDebugParameter(self.yin)
        z = p.readUserDebugParameter(self.zin)
        roll = p.readUserDebugParameter(self.rollId)
        pitch = p.readUserDebugParameter(self.pitchId)
        yaw = p.readUserDebugParameter(self.yawId)
        gripper_opening_length = p.readUserDebugParameter(
            self.gripper_opening_length_control
        )

        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def step(self, action, control_method="joint"):
        """
        Execute a step in the environment with the specified action.

        Parameters:
        - action (tuple): Desired control inputs for the robot. Format depends on `control_method`:
            - 'joint': (a1, a2, ..., a7, gripper_opening_length) - Joint angles and gripper length.
            - 'end': (x, y, z, roll, pitch, yaw, gripper_opening_length) - End effector pose and gripper length.
        - control_method (str): Method to control the robot. Options are 'joint' (joint space) or 'end' (end effector space).

        Returns:
        - Tuple: (observation, reward, done, info), where:
            - observation (dict): Current state information of the robot and environment.
            - reward (int): Reward signal based on the task progress.
            - done (bool): Whether the task is completed.
            - info (dict): Additional information, such as task states.
        """
        assert control_method in ("joint", "end")
        self.robot.move_ee(action[:-1], control_method)
        self.robot.move_gripper(action[-1])
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()

        reward = self.update_reward()
        done = True if reward == 1 else False
        info = dict(
            box_opened=self.box_opened,
            btn_pressed=self.btn_pressed,
            box_closed=self.box_closed,
        )
        return self.get_observation(), reward, done, info

    def update_reward(self):
        """
        Update the task reward based on the environment state.

        Checks specific conditions in the simulation to determine if the box has been opened, the button has been pressed,
        or the box has been closed, and updates the reward accordingly.

        Parameters:
        - None

        Returns:
        - reward (int): Task reward (1 if the task is completed, 0 otherwise).
        """
        reward = 0
        # if not self.box_opened:
        #     if p.getJointState(self.boxID, 1)[0] > 1.9:
        #         self.box_opened = True
        #         logger.info("Box opened!")
        # elif not self.btn_pressed:
        #     if p.getJointState(self.boxID, 0)[0] < -0.02:
        #         self.btn_pressed = True
        #         logger.info("Btn pressed!")
        # else:
        #     if p.getJointState(self.boxID, 1)[0] < 0.1:
        #         logger.info("Box closed!")
        #         self.box_closed = True
        #         reward = 1
        return reward

    def get_observation(self):
        """
        Collect the current state of the environment and robot.

        If a camera is available, captures RGB, depth, and segmentation images. Also retrieves the robot's joint states.

        Parameters:
        - None

        Returns:
        - observation (dict): Contains camera data (if available) and robot joint observations.
        """
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
        obs.update(self.robot.get_joint_obs())

        return obs

    def reset_box(self):
        """
        Reset the box to its initial state.

        Resets the position and orientation of the box to its initial values.

        Parameters:
        - None

        Returns:
        - None
        """
        initial_position = [0.0, 0.0, 0.0]
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(
            self.boxID, initial_position, initial_orientation
        )

    def reset(self):
        """
        Reset the environment to its initial state.

        Resets the robot and the box, and collects the initial observation.

        Parameters:
        - None

        Returns:
        - observation (dict): The initial state of the environment.
        """

        self.robot.reset()
        self.reset_box()
        return self.get_observation()

    def close(self):
        """
        Disconnect from the PyBullet physics server.

        Parameters:
        - None

        Returns:
        - None
        """
        p.disconnect(self.physicsClient)
