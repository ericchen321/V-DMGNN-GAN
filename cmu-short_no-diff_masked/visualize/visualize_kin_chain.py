import numpy as np
from processor.data_tools import expmap2rotmat
import matplotlib.pyplot as plt


def compute_affine_xform_mat(sample_index, time_index, joint_index, joint_rotmats, parents, bone_length):
    r"""
    Compute the 4D affine transformation matrix from joint's local coordinates
    to world coordinates.

    :return: 4, 4
    """
    # compute transform matrix from this joint to its parent
    # (counting only rotation here)
    xform_mat_4d = np.zeros([4, 4])
    xform_mat_4d[0:3, 0:3] = joint_rotmats[
        sample_index, time_index, joint_index, :, :
    ]
    xform_mat_4d[3, 3] = 1.0

    # compute transform matrix from parent to world
    if parents[joint_index] == -1:
        # base case: root joint to world
        pass
    else:
        # recursive case
        xform_mat_4d_last = compute_affine_xform_mat(
            sample_index,
            time_index,
            parents[joint_index],
            joint_rotmats,
            parents,
            bone_length
        )
        # include translation
        xform_mat_4d[0:3, 3] = np.array([0.0, 0.0, bone_length])
        # build hierarchy
        xform_mat_4d = np.matmul(xform_mat_4d_last, xform_mat_4d)
    return xform_mat_4d
        
def transform_along_kin_chain(sample_index, time_index, joint_index, joint_rotmats, parents, bone_length):
    r"""
    Compute one joint's position given joint rotation matrices,
    the kinematic tree and bone length.
    :param joint_rotmats: batch_size, T, J, 3, 3
    :param parents: J,
    :param bone_length: scalar

    :return: 3,
    """
    xform_mat_4d = compute_affine_xform_mat(
        sample_index,
        time_index,
        joint_index,
        joint_rotmats,
        parents,
        bone_length
    )
    joint_pos = np.matmul(xform_mat_4d, np.array([0.0, 0.0, 0.0, 1.0]))[0:3]
    return joint_pos

# exp map to joint pos
def compute_joint_pos(joint_exps, parents, bone_length):
    r"""
    Compute joint positions from joint rotations (in exponential maps).
    :param joint_exps: batch_size, T, J, 3
    :param parents: J,
    :param bone_length: scalar

    :return: batch_size, T, J, 3
    """
    batch_size, T, J, _ = joint_exps.shape

    # exp maps to rot matrices
    joint_rotmats = np.zeros([batch_size, T, J, 3, 3])
    for sample_index in range(batch_size):
        for time_index in range(T):
            for joint_index in range(J):
                joint_rotmats[sample_index, time_index, joint_index, :, :] = expmap2rotmat(
                    joint_exps[sample_index, time_index, joint_index, :]
                )

    # for each joint, compute its position by recursively transforming
    # along the kinematic chain
    joint_pos = np.zeros([batch_size, T, J, 3])
    for sample_index in range(batch_size):
        for time_index in range(T):
            for joint_index in range(J):
                joint_pos[sample_index, time_index, joint_index, :] = transform_along_kin_chain(
                    sample_index,
                    time_index,
                    joint_index,
                    joint_rotmats,
                    parents,
                    bone_length
                )
    return joint_pos


if __name__ == '__main__':
    # define the kinematic tree on the CMU dataset
    # (26 joints, not 38 joints)
    cmu_parents = np.array([
        8, 0, 1, 2, 8, 4, 5, 6, 9, 10, 11, -1, 11, 12, 10, 14, 15, 16, 17, 16, 11, 20, 21, 22, 23, 22
    ])

    # convert joint angles to joint positions in
    # the world frame
    encoder_inputs_4d = np.load(
        "/home/eric/eece571f/DMGNN/cmu-short_no-diff_masked/visualize/encoder_inputs.npy",
    )
    encoder_inputs_cartesian = compute_joint_pos(
        encoder_inputs_4d,
        cmu_parents,
        1.5
    )
    print(encoder_inputs_cartesian.shape)

    #visualize skeletons
    for time_id in range(encoder_inputs_cartesian.shape[1]):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        neighbor_link_ = [(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(1,9),(5,9),
                            (9,10),(10,11),(11,12),(12,13),(13,14),
                            (11,15),(15,16),(16,17),(17,18),(18,19),(17,20),
                            (12,21),(21,22),(22,23),(23,24),(24,25),(23,26)]
        neighbor_link = [(i-1,j-1) for (i,j) in neighbor_link_]
        encoder_inputs_sample_x_time_t = encoder_inputs_cartesian[4, time_id, :].reshape(-1, 3) # J, 3
        #print(encoder_inputs_sample_x_time_t.shape)
        #plot joints
        ax.scatter(
            encoder_inputs_sample_x_time_t[:, 0],
            encoder_inputs_sample_x_time_t[:, 1],
            encoder_inputs_sample_x_time_t[:, 2])
        #plot bones
        for bone in neighbor_link:
            ax.plot(encoder_inputs_sample_x_time_t[bone, 0],
                encoder_inputs_sample_x_time_t[bone, 1],
                encoder_inputs_sample_x_time_t[bone, 2],
                'ro-')
        #label joints
        for joint_index in range(encoder_inputs_sample_x_time_t.shape[0]):
            ax.text(encoder_inputs_sample_x_time_t[joint_index, 0],
                encoder_inputs_sample_x_time_t[joint_index, 1],
                encoder_inputs_sample_x_time_t[joint_index, 2],
                f"{joint_index}",
                color="blue")
        plt.savefig(f"skeletons/skeleton_<{time_id}>.png")
        plt.close(fig)
    print(f"plotted {encoder_inputs_cartesian.shape[1]} skeletons")