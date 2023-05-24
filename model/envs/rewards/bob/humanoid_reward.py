"""
Computing reward for Vanilla setup, constant target speed, gaussian kernels
"""
import numpy as np

from model.envs.rewards.utils.utils import *


class Reward:
    def __init__(self, cnt_timestep, num_joints, mimic_joints_index, reward_params):
        """
        :variable dt: control time step size
        :variable num_joints: number of joints
        :variable params: reward params read from the config file
        """
        self.dt = cnt_timestep
        self.num_joints = num_joints
        self.params = reward_params
        self.mimic_joints_index = mimic_joints_index

    def compute_reward(self, observation_raw, action_buffer, is_obs_fullstate, sample_retarget, end_effectors_pos):
        """
        Compute the reward based on observation (Vanilla Environment).

        :param observation_raw: current observation
        :param all_torques: torque records during the last control timestep
        :param action_buffer: history of previous actions
        :param is_obs_fullstate: flag to choose full state obs or not.

        :return: total reward, reward information (different terms can be passed here to be plotted in the graphs)
        """

        # test_data = {'observation_raw': observation_raw, 'dt': dt, 'num_joints': num_joints, 'params': params,
        #              'feet_status': feet_status, 'all_torques': all_torques, 'action_buffer': action_buffer,
        #              'is_obs_fullstate': is_obs_fullstate, 'joint_angles_default': joint_angles_default,
        #              'nominal_base_height': nominal_base_height}

        num_joints = self.num_joints
        dt = self.dt
        params = self.params

        observation = ObservationData(observation_raw, num_joints, is_obs_fullstate)

        # action_dot, action_ddot = calc_derivatives(action_buffer, dt, num_joints)

        # =======================
        # OURS MODEL coordinate   : Z FORWARD, X LEFT, Y UP
        # =======================

        # Root position reward
        desired_base_pos_xz = sample_retarget.q_fields.root_pos
        now_base_xz = observation.pos
        diff = np.linalg.norm(desired_base_pos_xz - now_base_xz)
        sigma_com = params.get("sigma_com", 0)
        weight_com = params.get("weight_com", 0)
        forward_vel_reward = weight_com * np.exp(-(diff**2) / (2.0 * sigma_com**2))

        # Root height reward
        height = observation.y
        desired_height = sample_retarget.q_fields.root_pos[1]
        diff_squere = (height - desired_height) ** 2
        sigma_height = params.get("sigma_height", 0)
        weight_height = params.get("weight_height", 0)
        height_reward = weight_height * np.exp(-diff_squere / (2.0 * sigma_height**2))

        # Root orientation reward
        (desired_yaw, desired_pitch, desired_roll) = sample_retarget.q_fields.root_rot
        roll_squere = (observation.roll - desired_roll) ** 2
        pitch_squere = (observation.pitch - desired_pitch) ** 2
        yaw_squere = (observation.yaw - desired_yaw) ** 2
        weight_root_ori = params.get("weight_root_ori", 0)
        sigma_root_ori = params.get("sigma_root_ori", 0)
        root_ori_reward = weight_root_ori * np.exp(
            -(roll_squere + pitch_squere + yaw_squere) / (6.0 * sigma_root_ori**2)
        )

        N = num_joints
        N_mimic = len(self.mimic_joints_index)
        # set_other = set(range(N)) - self.mimic_joints_index # joints that have no corresponding in motion clips

        # Unused joints smoothness reward
        # rest_joints_dot = action_dot[list(set_other)]
        # weight_smoothness1 = params.get("weight_smoothness1", 0)
        # sigma_smoothness1 = params.get("sigma_smoothness1", 0)
        # smoothness1_reward = weight_smoothness1 * np.exp(-np.sum(rest_joints_dot**2)/(2.0*(N-N_mimic)*sigma_smoothness1**2))

        # rest_joints_ddot = action_ddot[list(set_other)]
        # weight_smoothness2 = params.get("weight_smoothness2", 0)
        # sigma_smoothness2 = params.get("sigma_smoothness2", 0)
        # smoothness2_reward = weight_smoothness2 * np.exp(-np.sum(rest_joints_ddot**2)/(2.0*(N-N_mimic)*sigma_smoothness2**2))

        # motion_joints = sample_retarget.q_fields.joints
        # motion_joints_dot = sample_retarget.qdot_fields.joints
        # # Motion imitation reward
        # joint_angles = observation.joint_angles[list(self.mimic_joints_index)]
        # desired_angles = motion_joints[list(self.mimic_joints_index)]
        # diff = joint_angles - desired_angles
        # weight_joints = params.get("weight_joints", 0)
        # sigma_joints = params.get("sigma_joints", 0)
        # joints_reward = weight_joints * np.exp(-np.sum(np.square(diff))/(2.0*N_mimic*sigma_joints**2))

        # Leg reawrd (waiting for the end effector reward to take the place of it)
        # leg_joints_angles = observation.joint_angles[[1,4,7,10,13,16,2,5,8,11,14,17]]
        # desired_leg_angles = motion_joints[[1,4,7,10,13,16,2,5,8,11,14,17]]
        # diff = leg_joints_angles - desired_leg_angles
        # weight_legs = params.get("weight_legs", 0)
        # sigma_legs = params.get("sigma_legs", 0)
        # leg_reward = weight_legs * np.exp(-np.sum(np.square(diff))/(2.0*12*sigma_legs**2))

        # End effector reward
        # ignore x axis diff
        diff_lf = end_effectors_pos[0] - observation.lf
        diff_rf = end_effectors_pos[1] - observation.rf
        diff_lh = end_effectors_pos[2] - observation.lh
        diff_rh = end_effectors_pos[3] - observation.rh
        sum_diff_square = (
            np.sum(np.square(diff_lf))
            + np.sum(np.square(diff_rf))
            + np.sum(np.square(diff_lh))
            + np.sum(np.square(diff_rh))
        )

        weight_end_effectors = params.get("weight_joints_vel", 0)
        sigma_end_effectors = params.get("sigma_joints_vel", 0)
        end_effectors_reward = weight_end_effectors * np.exp(-sum_diff_square / (2.0 * 4 * sigma_end_effectors**2))

        # Joint velocities reward
        # joint_velocities = observation.joint_vel[list(self.mimic_joints_index)]
        # desired_velocities = motion_joints_dot[list(self.mimic_joints_index)]
        # diff = joint_velocities - desired_velocities
        # weight_joints_vel = params.get("weight_joints_vel", 0)
        # sigma_joints_vel = params.get("sigma_joints_vel", 0)
        # joints_vel_reward = weight_joints_vel * np.exp(-np.sum(np.square(diff))/(2.0*N_mimic*sigma_joints_vel**2))

        # =============
        # sum up rewards
        # =============
        smoothness1_reward = 0
        smoothness2_reward = 0
        smoothness_reward = params.get("weight_smoothness", 0) * (smoothness1_reward + smoothness2_reward)
        reward = forward_vel_reward + smoothness_reward + height_reward + root_ori_reward + end_effectors_reward

        info = {
            "forward_vel_reward": forward_vel_reward,
            "height_reward": height_reward,
            "root_ori_reward": root_ori_reward,
            "smoothness1_reward": smoothness1_reward,
            "smoothness2_reward": smoothness2_reward,
            "smoothness_reward": smoothness_reward,
            # "joints_reward": joints_reward,
            # "joints_vel_reward": joints_vel_reward,
            # "leg_reward": leg_reward,
            "end_effectors_reward": end_effectors_reward,
        }

        return reward, info

    def punishment(self, current_step, max_episode_steps):  # punishment for early termination
        penalty = self.params["weight_early_penalty"] * (max_episode_steps - current_step)
        return penalty
