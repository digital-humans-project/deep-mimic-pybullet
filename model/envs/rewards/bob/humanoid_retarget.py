import numpy as np
from scipy.spatial.transform import Rotation as R

class Retarget:
    def __init__(self, action_shape, joint_lower_limit, joint_upper_limit, joint_default_angle = None):

        self.joint_angle_limit_low = joint_lower_limit
        self.joint_angle_limit_high = joint_upper_limit
        self.joint_angle_default = joint_default_angle
        if self.joint_angle_default is None:
            self.joint_angle_default = np.array([ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,
                                                0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0 ])
        self.joint_scale_factors = np.maximum(abs(self.joint_angle_default - self.joint_angle_limit_low),
                                abs(self.joint_angle_default - self.joint_angle_limit_high))
        self.action_shape = action_shape
        
    def quart_to_rpy(self, q, mode):
        # q is in (w,x,y,z) format
        q_xyzw = list(q[1:])
        q_xyzw.append(q[0])
        r = R.from_quat(q_xyzw) 
        euler = r.as_euler(mode)
        return euler[0], euler[1], euler[2]
    
    def rescale_action(self, action):
        bound_action = np.minimum(np.maximum(action,self.joint_angle_limit_low),self.joint_angle_limit_high)
        scaled_action = (bound_action - self.joint_angle_default) / self.joint_scale_factors 
        return scaled_action
    
    def retarget_base_orientation(self, motion_clips_q):
        (yaw, pitch, roll)  = self.quart_to_rpy(motion_clips_q[3:7], 'yzx')
        return yaw, -pitch, roll

    def retarget_joint_angle(self, motion_clips_q, rescale = False):
        """Given a motion_clips orientation data, return a retarget action"""
        action = np.zeros(self.action_shape)

        (chest_z, chest_y, chest_x) = self.quart_to_rpy(motion_clips_q[7:11], 'zyx')
        (neck_z,  neck_y,  neck_x) = self.quart_to_rpy(motion_clips_q[11:15],'zyx')
        (r_hip_z, r_hip_x, r_hip_y) = self.quart_to_rpy(motion_clips_q[15:19],'zxy')
        (r_ankle_z, r_ankle_x, r_ankle_y) = self.quart_to_rpy(motion_clips_q[20:24],'zxy')
        (r_shoulder_z, r_shoulder_x, r_shoulder_y) = self.quart_to_rpy(motion_clips_q[24:28],'zxy')
        (l_hip_z, l_hip_x, l_hip_y) = self.quart_to_rpy(motion_clips_q[29:33],'zxy')
        (l_ankle_z, l_ankle_x, l_ankle_y) = self.quart_to_rpy(motion_clips_q[34:38],'zxy')
        (l_shoulder_z, l_shoulder_x, l_shoulder_y) = self.quart_to_rpy(motion_clips_q[38:42],'zxy')

        # chest - xyz euler angle 
        action[0] = -chest_z
        action[3] = chest_y
        action[6] = chest_x

        # neck - xyz euler angle 
        action[18] = -neck_z
        action[23] = neck_y
        action[26] = neck_x

        # shoulder - xzy euler angle 
        action[27] = -l_shoulder_z
        action[30] = l_shoulder_x
        action[33] = l_shoulder_y

        action[28] = -r_shoulder_z
        action[31] = r_shoulder_x
        action[34] = r_shoulder_y

        # ankle - xzy euler angle 
        action[13] = -l_ankle_z
        action[16] = l_ankle_x

        action[14] = -r_ankle_z
        action[17] = r_ankle_x            

        # hip - xzy euler angle 
        action[1] = -l_hip_z
        action[4] = l_hip_x
        action[7] = l_hip_y

        action[2] = -r_hip_z
        action[5] = r_hip_x
        action[8] = r_hip_y

        r_knee = motion_clips_q[19:20]
        r_elbow = motion_clips_q[28:29]
        l_knee = motion_clips_q[33:34]
        l_elbow = motion_clips_q[42:43]

        # elbow - revolute joint 
        action[36] = l_elbow
        action[37] = r_elbow

        # knee - revolute joint 
        action[10] = l_knee
        action[11] = r_knee

        if rescale is True:
            action = self.rescale_action(action)
        else:
            action = np.minimum(np.maximum(action,self.joint_angle_limit_low),self.joint_angle_limit_high)

        return action