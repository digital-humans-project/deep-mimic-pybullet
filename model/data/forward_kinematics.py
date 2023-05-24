import numpy as np
import urdf_parser_py.urdf as urdf
from scipy.spatial.transform import Rotation
import json

def extract_euler(quat, mode='XYZ'):
        r = Rotation.from_quat(quat)
        euler = r.as_euler(mode)
        return euler

class ForwardKinematics():
    """
    The forward kinematics is a class whose purpose is to extract the 3D position of any link (rigid body) of an agent.
    Currently, the implementation only supports URDF-formatted robot descriptions, from which the hierarchy of the agent is extracted.

    Attributes
    ----------
    file_path : str
        The path which contains the URDF description file of the agent.
    robot : urdf.Robot
        A "robot" object as described in the external library "urdf_parser_py".
        For more information, see https://github.com/ros/urdf_parser_py/blob/melodic-devel/src/urdf_parser_py/urdf.py

    data_map : dict, str -> Union[float|np.array|Rotation|None]
        For more information about the form of the dictionary, see "data_mapping(np.array)".
    """
    def __init__(self, urdf_file_path: str):
        """
        Parameters
        ----------
        urdf_file_path : str
            The path of the robot's URDF description.

        q : np.array
            A SINGLE line of data of the "Deep Mimic" paper.
        """
        self.__robot = urdf.URDF.from_xml_file(urdf_file_path)
        self.__data_map = None

    def load_motion_clip_frame(self, q: np.array):
        self.__data_map = self.__generate_data_mapping(q)

    def __generate_data_mapping(self, q : np.array):
        """
        Parameters
        ----------
        q : np.array
            A single time step sample from the Deep Mimic dataset.
            Currently, the mapping has been done manually (a.k.a. "by hand").

        Returns
        -------
        : dict, string -> str -> Union[float|np.array|Rotation|None]
            A mapping from each 1D joint to its corresponding scalar angle.
            str -> float: Joints. This is the most common form of key-value pairs.
            str -> np.array: only applies for the root translation. It is a 3D vector, i.e. dict['root_position'].shape() = (3,)
            str -> Rotation: only applies for root rotation. dict['root_rotation'] -> Rotation
            str -> None: applies for joints which could not be identified by Hu Jiangpeng (胡江鹏), a.k.a Marshall.
        """
        data_map = {}
        data_map['root_translation'] = np.array([q[1], q[2], q[3]])
        data_map['root_rotation'] = Rotation.from_quat([*q[5:8], q[4]])
        # The quaternions are provided in w, x, y, z, while ".as_euler()" accepts x, y, z, w format.
        data_map['root_chest_joint1'], data_map['root_chest_joint2'], data_map['root_chest_joint3'] = extract_euler([*q[9:12], q[8]])
        data_map['chest_neck_joint1'], data_map['chest_neck_joint2'], data_map['chest_neck_joint3'] = extract_euler([*q[13:16], q[12]])
        data_map['root_right_hip_joint1'], data_map['root_right_hip_joint2'], data_map['root_right_hip_joint3'] = extract_euler([*q[17:20], q[16]])
        data_map['right_knee'] = q[20]
        data_map['right_knee_right_ankle_joint1'], data_map['right_knee_right_ankle_joint2'],\
            data_map['right_knee_right_ankle_joint3'] = extract_euler([*q[22:25], q[21]])
        data_map['chest_right_shoulder_joint1'], data_map['chest_right_shoulder_joint2'], \
            data_map['chest_right_shoulder_joint3'] = extract_euler([*q[26:29], q[25]])
        data_map['right_elbow'] = q[29]
        data_map['root_left_hip_joint1'], data_map['root_left_hip_joint2'], data_map['root_left_hip_joint3'] = extract_euler([*q[31:34], q[30]])
        data_map['left_knee'] = q[34]
        data_map['left_knee_left_ankle_joint1'], data_map['left_knee_left_ankle_joint2'], data_map['left_knee_left_ankle_joint3'] = extract_euler([*q[36:39], q[35]])
        data_map['chest_left_shoulder_joint1'], data_map['chest_left_shoulder_joint2'], data_map['chest_left_shoulder_joint3'] = extract_euler([*q[40:43], q[39]])
        data_map['left_elbow'] = q[43]
        data_map['right_wrist'] = 0
        data_map['left_wrist_joint'] = 0
        data_map['root'] = 0

        return data_map
        
    def get_link_world_coordinates(self, link_str: str):
        """
        Parameters
        ----------
        link_str : str
            A string which expresses one of the links (rigid bodies) of the agent.

        Returns
        -------
        : np.array, np.array.shape() = (3,)
            A 3D array which expresses the position of the provided link's Center of Mass (CoM) in global coordinates.
        """
        # Currently, the implementation supports only agents constructed from the deep mimic dataset.
        # Therefore, the possible values for the parameter "link_str" are one of the following:
        # base_link base root
        # root_chest_link1 root_chest_link2
        # chest
        # chest_neck_link1 chest_neck_link2
        # neck
        # ==================== RIGHT SIDE ====================
        # root_right_hip_link1 root_right_hip_link2
        # right_hip 
        # right_knee
        # right_knee_right_ankle_link1 right_knee_right_ankle_link2
        # right_ankle
        # chest_right_shoulder_link1 chest_right_shoulder_link2
        # right_shoulder
        # right_elbow
        # right_wrist
        # ==================== LEFT SIDE ====================
        # left_hip root_left_hip_link1 root_left_hip_link2
        # left_knee
        # left_knee_left_ankle_link1 left_knee_left_ankle_link2
        # left_ankle
        # left_elbow
        # left_shoulder chest_left_shoulder_link1 chest_left_shoulder_link2
        # left_wrist

        # Initialize the to-be-returned-position to zero.
        local_position = np.zeros(3)


        # ALGORITHM EXPLANATION
        # STEP 1) Translate the CoM from the origin of the link's frame (which is identical to that of the joint's frame).
        # The skeletal structure of the agent is traversed "inwards" (from the end effector "link_str" towards the root).
        # STEP 2) Apply rotation of the translated CoM around the joint.
        # STEP 3) Move to the next (i.e. closer to the root) rigid body of the skeleton and apply  STEP 2.
        # STEP 4) Once the root is reached, apply the rotation & translation of the root (i.e. the rotation & translation of the whole agent)
        # Return the result.

        # STEP 1
        try:
            com_local_position = np.array(self.__robot.link_map[link_str].inertial.origin.xyz)
        except AttributeError:
            # In the Robot Operating System (ROS), no mention of origin implies that there is no translation of the CoM.
            # However, the (external) parser does not detect this. Therefore, it must be handled manually.
            com_local_position = np.zeros(3)


        # print(f'Initial rigid body: {link_str}')

        # print(f'local_position before = {local_position}\n'\
                # f'local_position after = {local_position + com_local_position}')
        local_position += com_local_position

        while link_str != 'base':
            # "joint" is the name of the joint which is responsible for connecting the current link "link_str"
            # and the previous link in the hierarchy "parent".  
            (joint, parent) = self.__robot.parent_map[link_str]

            # "angle" := scalar (float)  representing the amount of radians the CoM/end-effector is going to be rotated around "axis".
            angle = self.__data_map[joint]
            axis = self.__robot.joint_map[joint].axis
            # In the case of fixed joints (where axis is "None"), return any random rotation,
            # since its mapping angle is going to be zero (0).
            if axis is None:
                axis = np.array([1, 0, 0])
            else:
                axis = np.array(axis)

            # print(f'axis, angle = {axis, angle} of joint = {joint}')



            # STEP 2
            # Create a rotation matrix around the axis "axis" with angle "angle" and apply it to the (possibly translated) CoM.  
            r = Rotation.from_rotvec(angle * axis)
            rotation_matrix = r.as_matrix()
            local_position = rotation_matrix @ local_position # Apply rotation

            # STEP 3
            
            try:
                relative_to_parent_link_translation = np.array(self.__robot.joint_map[joint].origin.xyz)
            except AttributeError:
                # In the Robot Operating System (ROS), no mention of origin implies that
                # there is no frame translation between the parent and the child links.
                # However, the (external) parser does not detect this. Therefore, it must be handled manually.
                 relative_to_parent_link_translation = np.zeros(3)            
            local_position += relative_to_parent_link_translation

            # print(f'relative_to_parent_link_translation = {relative_to_parent_link_translation}')


            link_str = parent # Prepare for next iteration.
            # print(f'Rigid body updated to {link_str}')

        

        # Apply offset of 0.07.
        local_position += self.__robot.link_map['root'].inertial.origin.xyz

        # STEP 4
        local_position = self.__data_map['root_rotation'].apply(local_position)
        local_position += self.__data_map['root_translation']

        # The accumulations of local positions have formed the position in the global coordinate frame!
        return local_position
    

    def get_end_effectors_world_coordinates(self):
        # TODO: Check whether ankles can be added in the mapping...
        return np.array(
            [
                self.get_link_world_coordinates('left_ankle'),
                self.get_link_world_coordinates('right_ankle'),
                self.get_link_world_coordinates('left_wrist'),
                self.get_link_world_coordinates('right_wrist'),
            ])
    
def parse_motion_data(urdf_data_path, motion):
    end_effector_cood = []
    fk = ForwardKinematics(urdf_file_path=urdf_data_path)
    for i in range(motion.shape[0]):
        fk.load_motion_clip_frame(motion[i])
        end_effector_cood.append(fk.get_end_effectors_world_coordinates())
    end_effector_cood = np.array(end_effector_cood)
    return end_effector_cood



if __name__ == "__main__":

    
    
    # Manula data test
    # motion = [
    #      [        0.0333333015,        0.0000000000,        0.7577040000,        0.0000000000,       -0.9961860398,        0.0263161892,       -0.0021968997,        0.0831625275,        0.9891732485,       -0.0012968493,        0.0157765220,       -0.1458962099,        0.9575524872,       -0.0850737799,        0.1872322494,        0.2019895318,        0.9624490404,       -0.0077550214,       -0.1415960060,        0.2314784556,       -1.0795419930,        0.9644092584,        0.0987325715,       -0.1487828062,        0.3950136873,        0.9527796532,       -0.2560283448,        0.1562378638,       -0.0474357361,        1.5520426350,        0.8650128696,        0.0867517513,        0.1259648339,        0.4778699924,       -0.8698441741,        0.9978054552,       -0.0422917100,       -0.0396654391,       -0.0319740158,        0.8646200308,        0.4293653762,        0.1461038540,        0.2161740962,        2.2097567570],
    #      [        0.0333333015,        0.0071708444,        0.7627360000,       -0.0010782631,       -0.9966884035,        0.0196266459,       -0.0127152041,        0.0778803233,        0.9894542435,       -0.0047531444,        0.0100641608,       -0.1444175210,        0.9601754267,       -0.0864328594,        0.1836119737,        0.1920394591,        0.9644462354,       -0.0184400587,       -0.1529455748,        0.2147348931,       -1.0555873490,        0.9653237588,        0.0944503968,       -0.1428874431,        0.3970084816,        0.9539088588,       -0.2535200776,        0.1482817766,       -0.0616277052,        1.5403112662,        0.8723205719,        0.0754557007,        0.1339583807,        0.4641318878,       -0.8451495105,        0.9968932340,       -0.0526353394,       -0.0494114378,       -0.0314946164,        0.8738418463,        0.4183765245,        0.1310174601,        0.2102282965,        2.2068392952],
    #      [        0.0333333015,        0.0165471210,        0.7693890000,       -0.0040233937,       -0.9969061975,        0.0128295968,       -0.0235852159,        0.0738726770,        0.9900176675,       -0.0077922951,        0.0045254868,       -0.1406549611,        0.9635854119,       -0.0873562269,        0.1751337634,        0.1822092440,        0.9663925240,       -0.0292071827,       -0.1640127651,        0.1957862175,       -1.0213143186,        0.9664744149,        0.0919035346,       -0.1385599469,        0.3956580864,        0.9543634312,       -0.2526418060,        0.1395195811,       -0.0767909212,        1.5280580849,        0.8803353374,        0.0630091482,        0.1412631486,        0.4484242008,       -0.8155871236,        0.9958159574,       -0.0626401764,       -0.0588541001,       -0.0310319545,        0.8832629851,        0.4066816510,        0.1175508986,        0.2015894843,        2.2154124623]
    # ]

    motion = [
         [        0.0333333015,        0.0000000000,        0.7577040000,        0.0000000000,       -0.9961860398,        0.0263161892,       -0.0021968997,        0.0831625275,        0.9891732485,       -0.0012968493,        0.0157765220,       -0.1458962099,        0.9575524872,       -0.0850737799,        0.1872322494,        0.2019895318,        0.9624490404,       -0.0077550214,       -0.1415960060,        0.2314784556,       -1.0795419930,        0.9644092584,        0.0987325715,       -0.1487828062,        0.3950136873,        0.9527796532,       -0.2560283448,        0.1562378638,       -0.0474357361,        1.5520426350,        0.8650128696,        0.0867517513,        0.1259648339,        0.4778699924,       -0.8698441741,        0.9978054552,       -0.0422917100,       -0.0396654391,       -0.0319740158,        0.8646200308,        0.4293653762,        0.1461038540,        0.2161740962,        2.2097567570],
         [        0.0333333015,        0.0071708444,        0.7627360000,       -0.0010782631,       -0.9966884035,        0.0196266459,       -0.0127152041,        0.0778803233,        0.9894542435,       -0.0047531444,        0.0100641608,       -0.1444175210,        0.9601754267,       -0.0864328594,        0.1836119737,        0.1920394591,        0.9644462354,       -0.0184400587,       -0.1529455748,        0.2147348931,       -1.0555873490,        0.9653237588,        0.0944503968,       -0.1428874431,        0.3970084816,        0.9539088588,       -0.2535200776,        0.1482817766,       -0.0616277052,        1.5403112662,        0.8723205719,        0.0754557007,        0.1339583807,        0.4641318878,       -0.8451495105,        0.9968932340,       -0.0526353394,       -0.0494114378,       -0.0314946164,        0.8738418463,        0.4183765245,        0.1310174601,        0.2102282965,        2.2068392952],
         [        0.0333333015,        0.0165471210,        0.7693890000,       -0.0040233937,       -0.9969061975,        0.0128295968,       -0.0235852159,        0.0738726770,        0.9900176675,       -0.0077922951,        0.0045254868,       -0.1406549611,        0.9635854119,       -0.0873562269,        0.1751337634,        0.1822092440,        0.9663925240,       -0.0292071827,       -0.1640127651,        0.1957862175,       -1.0213143186,        0.9664744149,        0.0919035346,       -0.1385599469,        0.3956580864,        0.9543634312,       -0.2526418060,        0.1395195811,       -0.0767909212,        1.5280580849,        0.8803353374,        0.0630091482,        0.1412631486,        0.4484242008,       -0.8155871236,        0.9958159574,       -0.0626401764,       -0.0588541001,       -0.0310319545,        0.8832629851,        0.4066816510,        0.1175508986,        0.2015894843,        2.2154124623]
    ]

    # Print the numerical results of the first frame of the animation clip
    # Change the path to where your "deep-mimic" project is stored.
    fk = ForwardKinematics(r"C:\Users\kosta\Desktop\second_semester\digital_humans\final_project\deep-mimic\data\robots\deep-mimic\humanoid.urdf")
    fk.load_motion_clip_frame(motion[0])
    
    # # Include the following "for" loop in case you wish to see the results of the rest pose (where all limbs are fully extended).
    # for key, value in fk.data_map.items():
    #     if not key == 'root_rotation' and not key == 'root_translation':
    #         fk.data_map[key] = 0
    #     print(key, fk.data_map[key])

    print(f'All end-effect positions: {fk.get_end_effectors_world_coordinates()}')



    # Visualization of a whole data clip.
    # Change path to your path for the "humanoid.urdf". The file can be found int his project as well.
    urdf_data_path = r"C:\Users\kosta\Desktop\second_semester\digital_humans\final_project\deep-mimic\data\robots\deep-mimic\humanoid.urdf"
    motion_data_path = r'C:\Users\kosta\Desktop\second_semester\digital_humans\final_project\deep-mimic\data\deepmimic\motions\humanoid3d_jump.txt'
    with open(motion_data_path, "r") as json_file:
        data = json.load(json_file)
    motion = np.array(data["Frames"])
    end_effector_cood = parse_motion_data(urdf_data_path, motion)
    visualise_FK(end_effector_cood)

    # After visualizing the results, it seems that the feet are touching the floor when the forward kinematics elevation is at 0.1
    # If the height of the foot is taken into consideration, the actual difference should be even less (i.e. closer to 0).


    



