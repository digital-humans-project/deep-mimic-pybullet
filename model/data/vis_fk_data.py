import numpy as np
import matplotlib.pyplot as plt


def visualise_FK(arr):
    left_ankle_points = arr[:,0,:]
    right_ankle_points = arr[:,1,:]
    left_wrist_points = arr[:,2,:]
    right_wrist_points = arr[:,3,:]


    # Create a figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(left_ankle_points[:, 0], left_ankle_points[:, 1], left_ankle_points[:, 2], c='r', label='left_ankle')
    ax.scatter(right_ankle_points[:, 0], right_ankle_points[:, 1], right_ankle_points[:, 2], c='b', label='right_ankle')
    ax.scatter(left_wrist_points[:, 0], left_wrist_points[:, 1], left_wrist_points[:, 2], c='g', label='left_wrist')
    ax.scatter(right_wrist_points[:, 0], right_wrist_points[:, 1], right_wrist_points[:, 2], c='y', label='right_wrist')

    # # Plot set 2 points as blue dots
    # ax.scatter(set2_points[:, 0], set2_points[:, 1], set2_points[:, 2], c='b', label='Set 2')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End Effector Points')

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


# np_file_path = '/media/ankitaghosh/Data/ETH/digitalHumans/Project/DeepMimic/data/motions/humanoid3d_walk_endeff.npy'
# arr = np.load(np_file_path)
#visualise_FK(arr)