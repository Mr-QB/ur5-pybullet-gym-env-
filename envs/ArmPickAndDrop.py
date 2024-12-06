import time
import numpy as np
import pybullet as p
import pybullet_data
from tqdm import tqdm

from helper.utilities import Models, Camera, logger


class ArmPickAndDrop:

    SIMULATION_STEP_DELAY = 1 / 240.0

    def __init__(
        self,
        robot,
        models: Models,
        camera=None,
        vis=False,
        target_position_B=np.array([0.5, 0.5, 0.0]),
    ):
        """
        Initialize the environment for a robot to interact with a cluttered environment.

        Sets up the simulation environment, including gravity, the plane, and robot initialization.

        Parameters:
        - robot: The robot object that will be controlled in the environment.
        - models (Models): A set of models to be used in the environment.
        - camera: A camera object for visual feedback (optional).
        - vis (bool): If True, enables the graphical interface for visualization.
        - target_position_B (np.ndarray): The position of the target object in the robot's frame.
        """

        self.robot = robot
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera
        self.target_position_B = target_position_B

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
        self.draw_circle_at_target_position(self.target_position_B, radius=0.065)

        # For calculating the reward
        self.is_object_picked_up = False
        self.is_object_at_position_B = False
        self.is_object_dropped = False
        self.is_object_picked_up = False

    def draw_circle_at_target_position(
        self, target_position, radius=0.1, num_segments=30
    ):
        """
        Draw a circle at the target position in 3D space.

        This function approximates a circle by drawing line segments between points on the circle's perimeter.

        Parameters:
        - target_position (np.ndarray): The center coordinates of the circle [x, y, z].
        - radius (float): The radius of the circle.
        - num_segments (int): The number of line segments used to approximate the circle.

        Returns:
        - None
        """
        angle_step = 2 * np.pi / num_segments
        for i in range(num_segments):
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step

            x1 = target_position[0] + radius * np.cos(angle1)
            y1 = target_position[1] + radius * np.sin(angle1)
            z1 = target_position[2]

            x2 = target_position[0] + radius * np.cos(angle2)
            y2 = target_position[1] + radius * np.sin(angle2)
            z2 = target_position[2]

            p.addUserDebugLine(
                [x1, y1, z1], [x2, y2, z2], lineColorRGB=[1, 0, 0], lineWidth=2
            )

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
        done = self.check_task_done()
        info = {
            "is_object_picked_up": self.is_object_picked_up,
            "is_object_at_position_B": self.is_object_at_position_B,
            "is_object_dropped": self.is_object_dropped,
            "gripper_state": self.robot.get_gripper_state(),
            "robot_joint_states": self.robot.get_joint_obs(),
        }
        return self.get_observation(), reward, done, info

    def check_object_picked_up(self):
        """
        Check if the object has been successfully picked up by the robot.

        This function checks whether the robot's gripper has made contact with the object and if it is
        in a position that indicates the object has been picked up.

        Parameters:
        - None

        Returns:
        - bool: True if the object has been picked up, False otherwise.
        """
        object_position = p.getBasePositionAndOrientation(self.boxID)[0]
        gripper_position = p.getJointState(self.robot.id, self.robot.mimic_parent_id)[0]

        distance = np.linalg.norm(
            np.array(object_position) - np.array(gripper_position)
        )

        if distance < 0.01:
            self.is_object_picked_up = True
        else:
            self.is_object_picked_up = False

        return self.is_object_picked_up

    def check_object_at_position_B(self):
        """
        Check if the object is at the target position B.

        This function evaluates if the object has been successfully moved to position B, as defined by the environment.

        Parameters:
        - None

        Returns:
        - bool: True if the object is at position B, False otherwise.
        """

        object_position = p.getBasePositionAndOrientation(self.boxID)[0]

        distance_to_B = np.linalg.norm(
            np.array(object_position) - self.target_position_B
        )

        if distance_to_B < 0.005:
            self.is_object_at_position_B = True
        else:
            self.is_object_at_position_B = False

        return self.is_object_at_position_B

    def check_object_dropped(self):
        """
        Check if the object has been dropped or placed incorrectly.

        This function checks whether the object has fallen below a certain height or if the gripper is
        in the process of placing the object down correctly.

        Parameters:
        - None

        Returns:
        - bool: True if the object is dropped or out of bounds, False otherwise.
        """
        object_position = p.getBasePositionAndOrientation(self.boxID)[0]
        gripper_position = p.getJointState(self.robot.id, self.robot.mimic_parent_id)[0]

        if object_position[2] < 0.01:
            gripper_velocity = np.linalg.norm(
                np.array(gripper_position) - np.array(self.last_gripper_position)
            )

            if gripper_velocity < 0.01:
                self.is_object_dropped = False
            else:
                self.is_object_dropped = True
        else:
            self.is_object_dropped = False

        self.last_gripper_position = gripper_position

        return self.is_object_dropped

    def update_reward(self):
        """
        Update the reward based on the current state of the environment.

        This function calculates the current reward for the agent based on factors like object position,
        successful pick-up, and other task-specific criteria.

        Parameters:
        - None

        Returns:
        - float: The calculated reward for the current step in the environment.
        """
        reward = 0

        if not self.is_object_picked_up and self.check_object_picked_up():
            logger.info("Object picked up!")
            self.is_object_picked_up = True
            reward += 150

        if (
            self.is_object_picked_up
            and not self.is_object_at_position_B
            and self.check_object_at_position_B()
        ):
            logger.info("Object placed at position B!")
            self.is_object_at_position_B = True
            reward += 1000

        if self.check_object_dropped():
            logger.info("Object dropped!")
            reward -= 150

        if not self.is_object_at_position_B:
            reward -= 0.1
        hand_to_object_distance = self.get_hand_to_object_distance()
        if hand_to_object_distance > 0.0025:
            reward -= hand_to_object_distance * 0.1
        object_to_target_distance = self.get_object_to_target_distance()
        reward -= object_to_target_distance * 0.1
        return reward

    def get_hand_to_object_distance(self):
        hand_position = p.getJointState(self.robot.id, self.robot.mimic_parent_id)[0]
        object_position = p.getBasePositionAndOrientation(self.boxID)[0]
        distance = np.linalg.norm(np.array(hand_position) - np.array(object_position))
        return distance

    def get_object_to_target_distance(self):
        object_position = p.getBasePositionAndOrientation(self.boxID)[0]
        target_position = self.target_position_B
        distance = np.linalg.norm(np.array(object_position) - np.array(target_position))
        return distance

    def get_observation(self):
        """
        Collect the current state of the environment and robot.

        Captures RGB, depth, and segmentation images if a camera is available,
        and retrieves the robot's joint states.

        Returns:
        - numpy.ndarray: Flattened array combining all observation data.
        """
        obs = []

        obs.extend(self.robot.get_joint_angles())
        obs.extend(list(p.getBasePositionAndOrientation(self.boxID)[0]))
        obs.extend([self.get_hand_to_object_distance()])
        obs.extend([self.get_object_to_target_distance()])

        observation_array = np.array(obs)

        return observation_array

    def check_task_done(self):
        """
        Checks if the task is completed.

        Returns:
        - bool: True if the task is completed, False otherwise.
        """
        # and placed at the correct position
        if self.is_object_at_position_B and self.is_object_picked_up:
            return True
        return False

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
        self.draw_circle_at_target_position(self.target_position_B, radius=0.065)
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
