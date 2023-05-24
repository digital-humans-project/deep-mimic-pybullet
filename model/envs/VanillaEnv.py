import importlib.util
import sys
from importlib.machinery import SourceFileLoader

import numpy as np

from ..cmake_variables import PYLOCO_LIB_PATH
from .PylocoEnv import PylocoEnv

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)


from pylocogym.data.deep_mimic_bob_adapter import (
    BobMotionDataFieldNames,
    DeepMimicMotionBobAdapter,
)
from pylocogym.data.deep_mimic_motion import (
    DeepMimicMotion,
    DeepMimicMotionDataFieldNames,
)
from pylocogym.data.forward_kinematics import ForwardKinematics
from pylocogym.data.lerp_dataset import LerpMotionDataset
from pylocogym.data.loop_dataset import LoopKeyframeMotionDataset
from pylocogym.envs.rewards.bob.humanoid_reward import Reward


class VanillaEnv(PylocoEnv):
    def __init__(self, max_episode_steps, env_params, reward_params):
        sim_dt = 1.0 / env_params["simulation_rate"]
        con_dt = 1.0 / env_params["control_rate"]

        # if env_params['robot_model'] == "Dog":
        #     robot_id = 0
        # elif env_params['robot_model'] == "Go1":
        #     robot_id = 1
        # elif env_params['robot_model'] == "Bob":
        robot_id = 2

        loadVisuals = False
        super().__init__(pyloco.VanillaSimulator(sim_dt, con_dt, robot_id, loadVisuals), env_params, max_episode_steps)

        self._sim.lock_selected_joints = env_params.get("lock_selected_joints", False)
        self.enable_box_throwing = env_params.get("enable_box_throwing", False)
        self.box_throwing_interval = 100
        self.box_throwing_strength = 2
        self.box_throwing_counter = 0

        self.cnt_timestep_size = self._sim.control_timestep_size  # this should be equal to con_dt
        self.current_step = 0

        self.reward_params = reward_params
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.zeros(self.num_joints * 3)  # history of actions [current, previous, past previous]

        self.rng = np.random.default_rng(env_params.get("seed", 1))  # create a local random number generator with seed

        # Set maximum episode length according to motion clips
        self.clips_play_speed = reward_params["clips_play_speed"]  # play speed for motion clips
        self.clips_play_speed = reward_params["clips_play_speed"]  # play speed for motion clips
        self.clips_repeat_num = reward_params["clips_repeat_num"]  # the number of times the clip needs to be repeated
        self.initial_pose = np.concatenate(
            [
                np.array([0, self._sim.nominal_base_height, 0, 0, 0, 0]),
                self.joint_angle_default,
            ]
        )

        # Dataloader
        self.motion = DeepMimicMotion(reward_params["motion_clips_file_path"])
        self.dataset = DeepMimicMotionBobAdapter(
            reward_params["motion_clips_file_path"],
            self.num_joints,
            self.joint_angle_limit_low,
            self.joint_angle_limit_high,
            self.joint_angle_default,
            # initial_pose=self.initial_pose,
        )
        self.loop = LoopKeyframeMotionDataset(
            self.dataset, num_loop=self.clips_repeat_num, track_fields=[BobMotionDataFieldNames.ROOT_POS]
        )
        self.lerp = LerpMotionDataset(self.loop)

        self.motion_loop = LoopKeyframeMotionDataset(
            self.motion, num_loop=self.clips_repeat_num, track_fields=[DeepMimicMotionDataFieldNames.ROOT_POS]
        )
        self.motion_lerp = LerpMotionDataset(self.motion_loop)

        # Maximum episdode step
        self.max_episode_steps = (
            self.clips_repeat_num * (int(self.dataset.duration / self.cnt_timestep_size) - 1) / self.clips_play_speed
        )

        # Reward class
        self.reward_utils = Reward(
            self.cnt_timestep_size,
            self.num_joints,
            self.dataset.mimic_joints_index,
            reward_params,
        )

        # Forwards Kinematics class
        self.fk = ForwardKinematics(env_params["urdf_path"])

    def reset(self, seed=None, return_info=False, options=None, phase=None):
        # super().reset(seed=seed)  # We need this line to seed self.np_random
        self.current_step = 0
        self.box_throwing_counter = 0
        self.lerp.reset()  # reset dataloader
        self.motion_lerp.reset()  # reset

        # self.phase = self.sample_initial_state()
        if phase is None:
            self.phase = 0

        else:
            self.phase = phase
        self.initial_time = self.phase * self.dataset.duration

        (q_reset, qdot_reset) = self.get_initial_state(self.initial_time)
        self._sim.reset(q_reset, qdot_reset, self.initial_time)  # q, qdot include root's state(pos,ori,vel,angular vel)
        # self._sim.reset()

        observation = self.get_obs()
        self.sum_episode_reward_terms = {}
        self.action_buffer = np.concatenate(
            (self.joint_angle_default, self.joint_angle_default, self.joint_angle_default), axis=None
        )

        info = {"msg": "===Episode Reset Done!===\n"}
        return (observation, info) if return_info else observation

    def step(self, action: [np.ndarray]):
        # throw box if needed
        if self.enable_box_throwing and self.current_step % self.box_throwing_interval == 0:
            random_start_pos = (self.rng.random(3) * 2 - np.ones(3)) * 2  # 3 random numbers btw -2 and 2
            self._sim.throw_box(self.box_throwing_counter % 3, self.box_throwing_strength, random_start_pos)
            self.box_throwing_counter += 1

        # Accelerate or decelerate motion clips, usually deceleration
        # (clips_play_speed < 1 nomarlly)
        next_time = self._sim.get_time_stamp() + self.cnt_timestep_size
        next_t = (next_time - self.initial_time) * self.clips_play_speed + self.initial_time
        sample_retarget = self.lerp.eval(next_t)  # data after retargeting
        assert sample_retarget is not None
        
        # run simulation
        action_applied = self.scale_action(action) + sample_retarget.q_fields.joints
        self._sim.step(action_applied)
        observation = self.get_obs()

        # update variables
        self.current_step += 1
        self.action_buffer = np.roll(self.action_buffer, self.num_joints)  # moving action buffer
        self.action_buffer[0 : self.num_joints] = action_applied

        now_time = self._sim.get_time_stamp()
        now_t = (now_time - self.initial_time) * self.clips_play_speed + self.initial_time
        assert abs(now_t - next_t) < 1e-10

        """ Forwards and Inverse kinematics """
        # Load retargeted data
        # sample = self.motion_lerp.eval(now_t)  # original data for fk calculation
        # motion_clips_frame = np.concatenate([[0],sample.q])
        # self.fk.load_motion_clip_frame(motion_clips_frame)
        # end_effectors_pos = self.fk.get_end_effectors_world_coordinates()
        # x_pos = end_effectors_pos[:,0].copy()
        # z_pos = end_effectors_pos[:,2].copy()
        # end_effectors_pos[:,0] = -z_pos
        # end_effectors_pos[:,2] = x_pos
        # end_effectors_pos[:,1] -= 0.07

        # data_joints = sample_retarget.q
        # q_desired = self._sim.get_ik_solver_q(data_joints,
        #                                       end_effectors_pos[0,:],
        #                                       end_effectors_pos[1,:],
        #                                       end_effectors_pos[2,:],
        #                                       end_effectors_pos[3,:])

        end_effectors_raw = self._sim.get_fk_ee_pos(sample_retarget.q)
        end_effectors_pos = np.array(
            [end_effectors_raw[0], end_effectors_raw[2], end_effectors_raw[1], end_effectors_raw[3]]
        )
        # sample_retarget.q = q_desired

        # compute reward
        reward, reward_info = self.reward_utils.compute_reward(
            observation,
            self.action_buffer,
            self.is_obs_fullstate,
            sample_retarget,
            end_effectors_pos,
        )

        self.sum_episode_reward_terms = {
            key: self.sum_episode_reward_terms.get(key, 0) + reward_info.get(key, 0) for key in reward_info.keys()
        }

        # check if episode is done
        terminated, truncated, term_info = self.is_done(observation)
        done = terminated | truncated

        # punishment for early termination
        if terminated:
            reward -= self.reward_utils.punishment(self.current_step, self.max_episode_steps)

        info = {
            "is_success": truncated,
            "termination_info": term_info,
            "current_step": self.current_step,
            "action_applied": action_applied,
            "reward_info": reward_info,
            "TimeLimit.truncated": truncated,
            "msg": "=== 1 Episode Taken ===\n",
        }

        if done:
            mean_episode_reward_terms = {
                key: self.sum_episode_reward_terms.get(key, 0) / self.current_step for key in reward_info.keys()
            }
            info["mean_episode_reward_terms"] = mean_episode_reward_terms

        return observation, reward, done, info

    def filter_actions(self, action_new, action_old, max_joint_vel):
        # with this filter we have much fewer cases that joints cross their limits, but still not zero
        threshold = max_joint_vel * np.ones(self.num_joints)  # max angular joint velocity
        diff = action_new - action_old
        action_filtered = action_old + np.sign(diff) * np.minimum(np.abs(diff), threshold)
        return action_filtered
