import importlib.util
import sys

import gym
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation

from ..cmake_variables import PYLOCO_LIB_PATH

# importing pyloco
spec = importlib.util.spec_from_file_location("pyloco", PYLOCO_LIB_PATH)
pyloco = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = pyloco
spec.loader.exec_module(pyloco)

# default window size
DEFAULT_SIZE = 480


class PylocoEnv(gym.Env):
    def __init__(self, sim, env_params, max_episode_steps):
        super().__init__()

        assert sim is not None, "Oh no! pyloco sim is not yet created!"
        self._sim = sim
        self._viewer = None
        self._sim.motor_kp = env_params['motors_kp']
        self._sim.motor_kd = env_params['motors_kd']
        self._sim.motor_max_torque = env_params['max_torque']
        self.max_episode_steps = max_episode_steps

        # important note
        # metadata is required for video recording (used by video recoder wrapper)
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": env_params['control_rate'],
        }

        # joint information
        self.joint_angle_limit_low = self._sim.min_joint_angle
        self.joint_angle_limit_high = self._sim.max_joint_angle
        self.joint_angle_default = self._sim.nominal_joint_angle
        self.num_joints = self.joint_angle_default.size
        self.base_height_default = self._sim.nominal_base_height

        # action scaling factor
        self.joint_scale_factors = np.maximum(abs(self.joint_angle_default - self.joint_angle_limit_low),
                                              abs(self.joint_angle_default - self.joint_angle_limit_high))

        self.is_obs_fullstate = env_params.get("is_obs_fullstate", True)  # default to True

        if self.is_obs_fullstate:
            self.observation_low = np.concatenate((
                np.array([-20., 0., -20.]),  # base position: x = left, y = up, z = forward
                np.array([-np.pi, -np.pi, -np.pi]),  # base orientation (yaw, pitch, roll)
                self.joint_angle_limit_low - 0.1 * np.ones(self.num_joints),  # joint position
                np.array([-50., -50., -50.]),  # base linear velocity
                np.array([-50., -50., -50.]),  # base angular velocity
                -50. * np.ones(self.num_joints),  # joint velocity
                
                np.array([-20., 0., -20.]),  # lf pos 
                np.array([-20., 0., -20.]),  # rf pos
                np.array([-20., 0., -20.]),  # lh pos 
                np.array([-20., 0., -20.]),  # rh pos

                # np.float64(0.0),               # simulation_time
                np.float64(0.0)                # motion phase
            ), axis=None)

            self.observation_high = np.concatenate((
                np.array([20., 5., 20.]),  # base position: x = left, y = up, z = forward
                np.array([np.pi, np.pi, np.pi]),  # base orientation (yaw, pitch, roll)
                self.joint_angle_limit_high + 0.1 * np.ones(self.num_joints),  # joint position
                np.array([50., 50., 50.]),  # base linear velocity
                np.array([50., 50., 50.]),  # base angular velocity
                50. * np.ones(self.num_joints),  # joint velocity

                np.array([20., 5., 20.]),  # lf pos 
                np.array([20., 5., 20.]),  # rf pos
                np.array([20., 5., 20.]),  # lh pos 
                np.array([20., 5., 20.]),  # rh pos

                # np.inf,                      # simulation_time
                np.float64(1.0)              # motion phase
            ), axis=None)

            self.default_obs = np.concatenate((
                np.array([0., self.base_height_default, 0.]),  # base position: x = left, y = up, z = forward
                np.array([0., 0., 0.]),  # base orientation (yaw, pitch, roll)
                self.joint_angle_default,  # joint position
                np.array([0., 0., 0.]),  # base linear velocity
                np.array([0., 0., 0.]),  # base angular velocity
                np.zeros(self.num_joints),  # joint velocity

                np.array([0., 0., 0.]),  # lf pos 
                np.array([0., 0., 0.]),  # rf pos
                np.array([0., 0., 0.]),  # lh pos 
                np.array([0., 0., 0.]),  # rh pos

                # np.float64(0.0),          # simulation_time
                np.float64(0.0)           # motion phase
            ), axis=None)

        else:
            self.observation_low = np.concatenate((
                0.0,  # base y coordinate (height)
                np.array([-np.pi, -np.pi]),  # base roll and pitch
                self.joint_angle_limit_low - 0.1 * np.ones(self.num_joints),  # joint position
                np.array([-50., -50., -50.]),  # base linear velocity
                np.array([-50., -50., -50.]),  # base angular velocity
                -50. * np.ones(self.num_joints),  # joint velocity

                np.array([-20., 0., -20.]),  # lf pos 
                np.array([-20., 0., -20.]),  # rf pos
                np.array([-20., 0., -20.]),  # lh pos 
                np.array([-20., 0., -20.]),  # rh pos

                # np.float64(0.0),          # simulation_time
                np.float64(0.0)           # motion phase
            ), axis=None)

            self.observation_high = np.concatenate((
                5.,  # base y coordinate (height)
                np.array([np.pi, np.pi]),  # base roll and pitch
                self.joint_angle_limit_high + 0.1 * np.ones(self.num_joints),  # joint position
                np.array([50., 50., 50.]),  # base linear velocity
                np.array([50., 50., 50.]),  # base angular velocity
                50. * np.ones(self.num_joints),  # joint velocity'

                np.array([20., 5., 20.]),  # lf pos 
                np.array([20., 5., 20.]),  # rf pos
                np.array([20., 5., 20.]),  # lh pos 
                np.array([20., 5., 20.]),  # rh pos

                # np.inf,                      # simulation_time
                np.float64(1.0)              # motion phase
            ), axis=None)

            self.default_obs = np.concatenate((
                self.base_height_default,  # base y coordinate (height)
                np.array([0., 0.]),  # base roll and pitch
                self.joint_angle_default,  # joint position
                np.array([0., 0., 0.]),  # base linear velocity
                np.array([0., 0., 0.]),  # base angular velocity
                np.zeros(self.num_joints),  # joint velocity

                np.array([0., 0., 0.]),  # lf pos 
                np.array([0., 0., 0.]),  # rf pos
                np.array([0., 0., 0.]),  # lh pos 
                np.array([0., 0., 0.]),  # rh pos

                # np.float64(0.0),          # simulation_time
                np.float64(0.0)           # motion phase
            ), axis=None)

        self.observation_space = spaces.Box(
            low=self.observation_low,
            high=self.observation_high,
            shape=(len(self.observation_low),),
            dtype=np.float64)

        self.action_space = spaces.Box(
            low=-1.,
            high=1.,
            shape=(self.num_joints,),
            dtype=np.float64)  # normalized

        self.current_step = 0

    def is_done(self, observation):
        """ This function tells whether the episode is finished or not.
        Episode will finish if one of these conditions happen:
        1. max episode length is reached.
        2. robot falls down (low height or contact of body parts other than limbs to the ground).
        3. observation is out of range.
        """

        base_height = observation[1] if self.is_obs_fullstate else observation[0]

        # early termination
        if base_height < (self.base_height_default / 3) \
                or (not self.observation_space.contains(observation)) \
                or self._sim.is_robot_collapsed():
            if not self.observation_space.contains(observation):
                low_space = (observation >= self.observation_space.low)
                high_space = (observation <= self.observation_space.high)
                idx_low = np.where(low_space == False)
                idx_high = np.where(high_space == False)
                idx = np.concatenate((idx_low[0], idx_high[0]), axis=None)
                term_info = "out of range observation! index is:{}".format(idx)
            else:
                if self._sim.is_robot_collapsed():
                    term_info = "robot collapsed! Bad contact!"
                else:
                    term_info = "robot collapsed!"
            terminated = True
            truncated = False

        elif self.current_step > self.max_episode_steps:
            term_info = "reached max episode steps!"
            terminated = False
            truncated = True

        else:
            term_info = " "
            terminated = False
            truncated = False

        return terminated, truncated, term_info

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        if mode == "human":
            if self._viewer is None:
                # create full-screen viewer
                self._viewer = pyloco.Viewer(self._sim)

            # show plot for human
            self._viewer.show_plots = True
            self._viewer.render()
        elif mode == "rgb_array":
            if self._viewer is None:
                # create viewer with width and height
                self._viewer = pyloco.Viewer(self._sim, width, height)

            # hide plot for rgb_array
            self._viewer.show_plots = False
            self._viewer.render()
            size = self._viewer.width, self._viewer.height
            rgb_arr = np.zeros(3 * self._viewer.width * self._viewer.height, dtype=np.uint8)
            self._viewer.read_pixels(rgb_arr)
            data = np.array(rgb_arr, dtype=np.uint8).reshape((size[1], size[0], 3))
            data = np.flip(data, 0)
            return data
        else:
            pass

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            del self._viewer
            self._viewer = None

    def get_obs(self):
        q = self._sim.get_q()
        qdot = self._sim.get_qdot()
        if self.is_obs_fullstate:
            obs = np.concatenate((q, qdot), axis=None)
        else:
            obs = self.get_reduced_obs(q, qdot)

        # add end_effector pos
        end_effector = self.get_feet_status()
        lfoot_pos = end_effector.pos[0]
        lhand_pos = end_effector.pos[1]
        rfoot_pos = end_effector.pos[2]
        rhand_pos = end_effector.pos[3]

        obs = np.concatenate((obs, lfoot_pos, rfoot_pos, lhand_pos, rhand_pos), axis=None)

        # add simulation time
        now_time = self._sim.get_time_stamp()
        dt_actual = (now_time-self.initial_time)*self.clips_play_speed
        now_phase, _ = np.modf(self.phase + (dt_actual/self.dataset.duration))
        obs = np.concatenate((obs, now_phase), axis=None)

        return obs

    def get_reduced_obs(self, q, qdot):
        """ Get reduced observation (only observable states):
                [base height, base pitch, base roll, joint angles,
                base velocity in body frame, base angular velocity in body frame, joint speed]
        """

        pos = q[0:3]  # x = left, y = up, z = forward
        ori = q[3:6]  # yxz Euler angles (yaw, pitch, roll)
        joint_ang = q[6:]

        lin_vel = qdot[0:3]
        ang_vel = qdot[3:6]
        joint_vel = qdot[6:]

        # compute body frame velocity and angular velocity
        # R_BW = R_WB.transpose()
        R = Rotation.from_euler('yxz', -ori)
        local_lin_vel = R.apply(lin_vel)
        local_ang_vel = R.apply(ang_vel)

        obs = np.concatenate((pos[1], ori[1:3], joint_ang, local_lin_vel, local_ang_vel, joint_vel), axis=None)
        return obs

    def scale_action(self, action):
        # the action space now doesn't fill the joint limits due to asymmetry
        return np.minimum(
            np.maximum(action * self.joint_scale_factors + self.joint_angle_default, self.joint_angle_limit_low),
            self.joint_angle_limit_high)
        # return action * self.joint_scale_factors + (self.joint_angle_limit_low + self.joint_angle_limit_high) / 2

    def get_initial_state(self, initial_time):
        # Get desired root state, joint state according to phase
        now_t = initial_time

        # Load retargeted data
        sample_retarget = self.lerp.eval(now_t) # data after retargeting
        # sample = self.motion_lerp.eval(now_t)  # original data for fk calculation
        
        # motion_clips_frame = np.concatenate([[0],sample.q])
        # self.fk.load_motion_clip_frame(motion_clips_frame)
        # end_effectors_pos = self.fk.get_end_effectors_world_coordinates()
        # x_pos = end_effectors_pos[:,0].copy()
        # end_effectors_pos[:,2] = x_pos
        # end_effectors_pos[:,1] -= 0.07

        # q_desired = self._sim.get_ik_solver_q(sample_retarget.q,
        #                                       end_effectors_pos[0,:],
        #                                       end_effectors_pos[1,:],
        #                                       end_effectors_pos[2,:],
        #                                       end_effectors_pos[3,:])
        
        q_reset = sample_retarget.q
        qdot_reset = sample_retarget.qdot
        # qdot_reset = np.zeros(len(self.joint_angle_default) + 6)
        
        # for debug useage
        # q_reset = np.zeros(len(self.joint_angle_default) + 6)
        # q_reset[0:3] = np.array([0,0.9,0]) 
        # q_reset[3:6] = np.array([0,0,0])
        # qdot_reset = np.zeros(len(self.joint_angle_default) + 6)
        # qdot_reset[0:3] = np.array([0,0,0])
        # qdot_reset[3:6] = np.array([0,0,0]) # angular velocity: [Y, X, Z]

        return q_reset, qdot_reset

    def sample_initial_state(self):
        # Random sample phase from [0,1)
        seed = np.random.random_sample()
        # self.phase = np.abs(np.random.normal(loc=0.0, scale=0.2, size=None))
        # self.phase, _ = np.modf(self.phase)
        if seed < 0.7:
            return 0
        if seed < 0.8:
            return 0.2
        if seed < 0.9:
            return 0.4
        if seed < 0.95:
            return 0.6
        return 0.8

    def get_feet_status(self):
        status = FeetStatus()
        status.pos = self._sim.get_feet_pos()
        status.vel = self._sim.get_feet_vel()
        status.isSwing = 1 - self._sim.is_feet_contact()
        return status


class FeetStatus:
    pos = 0
    vel = 0
    isSwing = 0
