from processor.data_tools import _some_variables, fkl
import numpy as np
import matplotlib.pyplot as plt
from map_38_joints_to_26_joints import joints_26, joints_38

parent_38, offset_38, posInd_38, expmapInd_38 = _some_variables()

def find_nearest_not_removed_joint(joint_38_index, parent_38, joints_38):
    r"""
    Find the index of the nearest (parent) joint to `joint_38_index`
    that is not removed in the 26-joint model.
    """
    # if can't find one, return -1
    nearest_joint_index = -1

    # Trace along the branch. Find the nearest parent not removed
    parent_id = parent_38[joint_38_index]
    while parent_id != -1 and joints_38[parent_id] == -1:
        parent_id = parent_38[parent_id]
    if joints_38[parent_id] != -1:
        # can find a parent along the branch not removed
        nearest_joint_index = parent_id
    else:
        # found all parents along the branch removed
        # then use the smallest not-removed joint
        for smallest_id in range(0, joint_38_index, 1):
            if joints_38[smallest_id] != -1:
                nearest_joint_index = smallest_id
                break

    return nearest_joint_index

# set parent
def get_parent_index_in_38j_model(joint_38_index, parent_38, joints_38):
    r"""
    Get the index of joint `joint_38_index`'s parent
    in the 38-joints model. Consider joint removals
    in the 26-joints model.
    """
    parent_38_index = parent_38[joint_38_index]
    # if parent is removed in the 26j-model
    if joints_38[parent_38_index] == -1:
        # use the nearest not-removed joint as the parent
        # if itself is, make it root
        parent_38_index = find_nearest_not_removed_joint(
            joint_38_index,
            parent_38,
            joints_38
        )

    # otherwise just return the index as defined in parent_38
    return parent_38_index

parent_26 = np.zeros([26, ], dtype=np.int32)
for i in range(len(joints_26)):
    # get the index in the 38-joint model
    joint_38_index = joints_26[i]
    # get the parent's index in the 38-joint model
    parent_38_index = get_parent_index_in_38j_model(
        joint_38_index,
        parent_38,
        joints_38
    )
    # get the parent's index in the 26-joint model
    if parent_38_index == -1:
        parent_26_index = -1
    else:
        parent_26_index = joints_38[parent_38_index]
    # print(parent_26_index)
    parent_26[i] = parent_26_index

# set offset
# print(offset_38)
offset_26 = offset_38[joints_26]

# set posInd
posInd_26 = np.array(posInd_38)[joints_26]
# print(posInd_26)

# set expmapInd
expmapInd_26 = np.split(np.arange(0,78), 26)

# for i in range(len(posInd)):
#     print(expmapInd[i])

# print(len(expmapInd))
# for i in range(len(expmapInd)):
#     print(expmapInd[i])

encoder_inputs_4d = np.load(
    "/home/eric/eece571f/DMGNN/cmu-short_no-diff_masked/visualize/encoder_inputs_26_joints_walking.npy",
)
encoder_inputs_3d = encoder_inputs_4d.reshape(
    encoder_inputs_4d.shape[0],
    encoder_inputs_4d.shape[1],
    -1
)
print(f"encoder_inputs_3d has shape {encoder_inputs_3d.shape}")

sample_id = 0
for time_id in range(encoder_inputs_3d.shape[1]):
    xyz = fkl(encoder_inputs_3d[sample_id, time_id, :], parent_26, offset_26, posInd_26, expmapInd_26)
    # print(xyz.shape)
    xyz_2d = xyz.reshape(-1, 3)
    # for i in range(xyz_2d.shape[0]):
    #     print(xyz_2d[i])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    #print(encoder_inputs_sample_x_time_t.shape)
    #plot joints
    ax.scatter(
        xyz_2d[:, 2],
        xyz_2d[:, 1],
        xyz_2d[:, 0])
    #plot bones
    neighbor_link_partition = {
        "left_arm": [
            (10, 14), (14, 15), (15, 16), (16, 19), (16, 17), (17, 18)
        ],
        "right_arm": [
            (11, 20), (20, 21), (21, 22), (22, 23), (23, 24), (23, 25)
        ],
        "torso": [
            (8, 9), (9, 10), (10, 11)
        ],
        "head": [
            (11, 12), (12, 13)
        ],
        "left_leg": [
            (0, 8), (0, 1), (1, 2), (2, 3)
        ],
        "right_leg": [
            (4, 8), (4, 5), (5, 6), (6, 7)
        ]
    }
    for part_name, part_links in neighbor_link_partition.items():
        if part_name == "left_arm" or part_name == "right_arm":
            link_color = "red"
            link_width = 2
        elif part_name == "torso":
            link_color = "cyan"
            link_width = 4
        elif part_name == "head":
            link_color = "purple"
            link_width = 2
        else:
            link_color = "green"
            link_width = 2
        for bone in part_links:
            ax.plot(xyz_2d[bone, 2],
                xyz_2d[bone, 1],
                xyz_2d[bone, 0],
                color = link_color,
                linewidth = link_width)
    #label joints
    for joint_index in range(xyz_2d.shape[0]):
        ax.text(xyz_2d[joint_index, 2],
            xyz_2d[joint_index, 1],
            xyz_2d[joint_index, 0],
            f"{joint_index}",
            fontsize = 12,
            color="blue")
    plt.savefig(f"skeletons_26_joints_walking_fkl/skeleton_<{time_id}>.png")
    plt.close(fig)
print(f"plotted {encoder_inputs_3d.shape[1]} skeletons")