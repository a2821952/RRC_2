"""Example policies, demonstrating how to control the robot."""
import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
import ipdb
import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils
from trifinger_object_tracking.py_lightblue_segmenter import segment_image
import copy

class PointAtTrajectoryPolicy:
    """Dummy policy which just points at the goal positions with one finger.

    This is a simple example policy that does not even attempt to pick up the
    cube but simple points at the goal positions where the cube should be using
    one finger.
    """

    def __init__(self, action_space, trajectory):
        self.action_space = action_space
        self.trajectory = trajectory

        robot_properties_path = get_package_share_directory(
            "robot_properties_fingers"
        )
        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
            "trifingerpro"
        )
        finger_urdf_path = os.path.join(
            robot_properties_path, "urdf", urdf_file
        )
        self.kinematics = trifinger_simulation.pinocchio_utils.Kinematics(
            finger_urdf_path,
            [
                "finger_tip_link_0",
                "finger_tip_link_120",
                "finger_tip_link_240",
            ],
        )

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array([0, 1.5, -2.7] * 3)

    def clip_to_space(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def predict(self, observation, t):
        # in the first few steps keep the target position fixed to move to the
        # initial position (to avoid collisions between the fingers)

        if t > 500:
            goal_pos = observation["desired_goal"]

            # get joint positions for finger 0 to move its tip to the goal position
            new_joint_pos, err = self.kinematics.inverse_kinematics_one_finger(
                0,
                goal_pos,
                observation["robot_observation"]["position"],
            )

            # slowly update the target position of finger 0 (leaving the other two
            # fingers unchanged)
            alpha = 0.01
            self.joint_positions[:3] = (
                alpha * new_joint_pos[:3] + (1 - alpha) * self.joint_positions[:3]
            )

            # make sure to not exceed the allowed action space
            self.joint_positions = self.clip_to_space(self.joint_positions)

        # make sure to return a copy, not a reference to self.joint_positions
        return np.array(self.joint_positions)


def trans_location(x, y):
    real_x = 0.0019 * x - 0.27
    real_y = -0.00235 * y + 0.306
    return real_x, real_y

class PointAtDieGoalPositionsPolicy:
    """Dummy policy which just points at the goal positions with one finger.

    This is a simple dummy policy that moves one finger around, pointing at the
    positions at which the dice should be placed.

    Note: This is mostly a duplicate of ``PointAtTrajectoryPolicy`` with only
    some small modifications for the different goal format.
    """

    def __init__(self, action_space, goal):
        self.action_space = action_space
        self.goal = goal
        self.idx = 0
        self.last_idx_update = 0

        robot_properties_path = get_package_share_directory(
            "robot_properties_fingers"
        )
        urdf_file = trifinger_simulation.finger_types_data.get_finger_urdf(
            "trifingerpro"
        )
        finger_urdf_path = os.path.join(
            robot_properties_path, "urdf", urdf_file
        )
        self.kinematics = trifinger_simulation.pinocchio_utils.Kinematics(
            finger_urdf_path,
            [
                "finger_tip_link_0",
                "finger_tip_link_120",
                "finger_tip_link_240",
            ],
        )

        # initial joint positions (lifting the fingers up)
        self.joint_positions = np.array([0, 1.5, -2.7] * 3)

    def clip_to_space(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)

    def predict(self, observation, t):
        # in the first few steps keep the target position fixed to move to the
        # initial position (to avoid collisions between the fingers)
        # ipdb.set_trace()

        if t == 0:
            self.his_goal = observation['achieved_goal'][1]
            # ipdb.set_trace()

        if t > 500:
            # every 2000 steps move to the next position
            if t - self.last_idx_update > 1000:
                time_idx = self.last_idx_update // 1000
                # goal_pos = np.array([time_idx * 0.03 - 0.15, 0.15 * (-1) ** (time_idx % 2), 0.03])
                goal_pos = np.array([0.15 * (-1) ** (time_idx % 2+1), time_idx * 0.03 - 0.15, 0.03])
                print(goal_pos)


                self.idx = (self.idx + 1) % len(self.goal)
                self.last_idx_update += 1000
                print("Switch to position #{}".format(self.idx), flush=True)
                # test = copy.deepcopy(observation['desired_goal'][0])
                # test[observation['desired_goal'][0] == observation['achieved_goal'][0]] = 0
                #
                # test1 = copy.deepcopy(observation['achieved_goal'][0])
                # test1[observation['desired_goal'][0] == observation['achieved_goal'][0]] = 0

                test1 = copy.deepcopy(observation['desired_goal'][1])
                test1[observation['desired_goal'][1] == observation['achieved_goal'][1]] = 0

                test2 = copy.deepcopy(observation['achieved_goal'][1])
                test2[observation['desired_goal'][1] == observation['achieved_goal'][1]] = 0

                # test4 = copy.deepcopy(observation['desired_goal'][2])
                # test4[observation['desired_goal'][2] == observation['achieved_goal'][2]] = 0
                #
                # test5 = copy.deepcopy(observation['achieved_goal'][2])
                # test5[observation['desired_goal'][2] == observation['achieved_goal'][2]] = 0

                result1 = (test1 + 0.5 * test2)/255
                # result2 = (test2 + 0.5 * test3)/255
                # result3 = (test4 + 0.5 * test5)/255

                if self.last_idx_update % 2000 == 1000:
                    his_data = self.his_goal / 255
                    now_data = observation['achieved_goal'][1]/255
                    diff = (now_data!=his_data).sum(1).argmax()
                    print(diff)
                    self.his_goal = observation['achieved_goal'][1]



            time_idx = self.last_idx_update // 1000
            # goal_pos = np.array([time_idx * 0.03 - 0.15, 0.15 * (-1) ** (time_idx % 2), 0.03])
            goal_pos = np.array([0.15 * (-1) ** (time_idx % 2+1), time_idx * 0.03 - 0.15, 0.03])


            # goal_pos = self.goal[self.idx] + np.array([0, 0, 0.01])







            # get joint positions for finger 0 to move its tip to the goal position
            new_joint_pos, err = self.kinematics.inverse_kinematics_one_finger(
                0,
                goal_pos,
                observation["robot_observation"]["position"],
                max_iterations=7,
            )

            # slowly update the target position of finger 0 (leaving the other two
            # fingers unchanged)
            alpha = 0.01

            # if (new_joint_pos[:3] < self.action_space.high[:3]).all() and (new_joint_pos[:3] > self.action_space.low[:3]).all():

            self.joint_positions[:3] = (
                alpha * new_joint_pos[:3] + (1 - alpha) * self.joint_positions[:3]
            )




            new_joint_pos, err = self.kinematics.inverse_kinematics_one_finger(
                1,
                goal_pos,
                observation["robot_observation"]["position"],
                max_iterations=7,
            )

            # if (new_joint_pos[3:6] < self.action_space.high[3:6]).all() and (
            #         new_joint_pos[3:6] > self.action_space.low[3:6]).all():
            self.joint_positions[3:6] = (
                alpha * new_joint_pos[3:6] + (1 - alpha) * self.joint_positions[3:6]
            )





            new_joint_pos, err = self.kinematics.inverse_kinematics_one_finger(
                2,
                goal_pos,
                observation["robot_observation"]["position"],
                max_iterations=7,
            )


            # if (new_joint_pos[6:] < self.action_space.high[6:]).all() and (
            #         new_joint_pos[6:] > self.action_space.low[6:]).all():
            self.joint_positions[6:] = (
                alpha * new_joint_pos[6:] + (1 - alpha) * self.joint_positions[6:]
            )


            # make sure to not exceed the allowed action space
            self.joint_positions = self.clip_to_space(self.joint_positions)

        # make sure to return a copy, not a reference to self.joint_positions
        return np.array(self.joint_positions)




