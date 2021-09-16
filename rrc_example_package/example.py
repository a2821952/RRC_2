"""Example policies, demonstrating how to control the robot."""
import os

import numpy as np
from ament_index_python.packages import get_package_share_directory
import ipdb
import trifinger_simulation.finger_types_data
import trifinger_simulation.pinocchio_utils
from trifinger_object_tracking.py_lightblue_segmenter import segment_image
import copy
import cv2
import random

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


    def render_figure(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()
    def predict(self, observation, t):
        # in the first few steps keep the target position fixed to move to the
        # initial position (to avoid collisions between the fingers)

        s_r = observation['achieved_goal'][1]
        s_h = observation['desired_goal'][1]
        if t == 0:
            self.his_goal = observation['achieved_goal'][1]

            self.point_x = [7, 7]
            self.point_y = [7, 7]
            self.his_point = [0,0]

        goal_pos = np.array([0, 0.1, 0.15])
        cycle = 2000

        if t >= 500:

            # every 2000 steps move to the next position

            if t - self.last_idx_update == 500:
                self.last_idx_update += cycle
                print("Switch to position #{}".format(self.last_idx_update // cycle), flush=True)

                test1 = copy.deepcopy(observation['desired_goal'][1])
                test1[observation['desired_goal'][1] == observation['achieved_goal'][1]] = 0
                fix1 = np.zeros([15, 15])
                for i in range(15):
                    for j in range(15):
                        fix1[i, j] = sum(sum(test1[i * 18:(i + 1) * 18, j * 18:(j + 1) * 18] / 255))

                test2 = copy.deepcopy(observation['achieved_goal'][1])
                test2[observation['desired_goal'][1] == observation['achieved_goal'][1]] = 0
                fix2 = np.zeros([15, 15])
                for i in range(15):
                    for j in range(15):
                        fix2[i, j] = sum(sum(test2[i * 18:(i + 1) * 18, j * 18:(j + 1) * 18] / 255))


                # mean1 = np.zeros([15, 15])
                # for i in range(15):
                #     for j in range(15):
                #         mean1[i, j] = sum(sum(fix1[i:i+3,j:j+3]))
                #
                # mean2 = np.zeros([15, 15])
                # for i in range(15):
                #     for j in range(15):
                #         mean2[i, j] = sum(sum(fix2[i:i+3,j:j+3]))

                mean1 = fix1
                mean2 = fix2
                # ipdb.set_trace()
                p = 0
                while p == 0:
                    max1 = mean1.argmax()
                    ind1 = np.unravel_index(max1, mean1.shape)

                    max2 = mean2.argmax()
                    ind2 = np.unravel_index(max2, mean2.shape)

                    if (ind1[0] - ind2[0]) <= 5 and (ind1[1] - ind2[1]) <= 5 and abs(ind1[0] - self.his_point[0])>1:
                        p = 1

                    if random.random() < 0.5:
                        mean1[ind1[0],ind1[1]] = 0
                    else:
                        mean2[ind2[0],ind2[1]] = 0
                    self.his_point[0] = ind1[0]


                result1 = (test1 + 0.5 * test2) / 255
                self.point_x = [ind1[0],ind1[1]]
                self.point_y = [ind2[0],ind2[1]]
                # print(self.point_x, self.point_y)
                self.his_point = copy.deepcopy(self.point_x)



            point_x = self.point_x
            point_y = self.point_y

            first_point = copy.deepcopy(point_x)
            if point_y[0] > point_x[0]:
                first_point[0] -= 1
            else:
                first_point[0] += 1

            real_x, real_y = trans_location(first_point[0] * 18, first_point[1] * 18)
            real_x = np.clip(real_x, -0.17, 0.17)
            real_y = np.clip(real_y, -0.17, 0.17)
            if t % cycle < 300:
                goal_pos = np.array([real_x, real_y, 0.15])
            elif t % cycle < 500:
                goal_pos = np.array([real_x, real_y, 0.01])

            elif t % cycle < 900:
                real_x, real_y = trans_location((point_y[0]) * 18, first_point[1] * 18)
                real_x = np.clip(real_x, -0.17, 0.17)
                real_y = np.clip(real_y, -0.17, 0.17)
                goal_pos = np.array([real_x, real_y, 0.01])
            elif t % cycle < 1100:
                if point_y[1] < point_x[1]:
                    relate_y = 1
                else:
                    relate_y = -1
                real_x, real_y = trans_location((point_y[0]) * 18, (first_point[1] + relate_y) * 18)
                real_x = np.clip(real_x, -0.17, 0.17)
                real_y = np.clip(real_y, -0.17, 0.17)
                goal_pos = np.array([real_x, real_y, 0.08])

            elif t % cycle < 1300:
                if point_y[1] < point_x[1]:
                    relate_y = 1
                else:
                    relate_y = -1
                real_x, real_y = trans_location((point_y[0]) * 18, (first_point[1] + relate_y) * 18)
                real_x = np.clip(real_x, -0.17, 0.17)
                real_y = np.clip(real_y, -0.17, 0.17)
                goal_pos = np.array([real_x, real_y, 0.01])

            elif t % cycle < 1700:
                real_x, real_y = trans_location((point_y[0]) * 18, point_y[1] * 18)
                real_x = np.clip(real_x, -0.17, 0.17)
                real_y = np.clip(real_y, -0.17, 0.17)
                goal_pos = np.array([real_x, real_y, 0.01])

            # if self.last_idx_update % 2000 == 1000:
            #     his_data = self.his_goal / 255
            #     now_data = observation['achieved_goal'][1]/255
            #     self.his_goal = observation['achieved_goal'][1]



        # get joint positions for finger 0 to move its tip to the goal position
        new_joint_pos, err = self.kinematics.inverse_kinematics_one_finger(
            0,
            goal_pos,
            observation["robot_observation"]["position"],
            max_iterations=7,
        )

        # slowly update the target position of finger 0 (leaving the other two
        # fingers unchanged)
        alpha = 0.05

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




