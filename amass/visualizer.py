import imageio
from tqdm import tqdm
import numpy as np
import os
import torch
import trimesh
from body_visualizer.tools.vis_tools import colors, imagearray2file
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.mesh.sphere import points_to_spheres
from body_visualizer.tools.vis_tools import show_image
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import matplotlib.pyplot as plt


def gif_writer(output_dir, gif_name, num_frames, img_pre_fix='frame'):
    with imageio.get_writer(output_dir + '/' + gif_name + '.gif', mode='I') as writer:
        for i in tqdm(range(num_frames)):
            image = imageio.imread(output_dir + '/{}{}.png'.format(img_pre_fix, i))
            writer.append_data(image)


def generate_gif_animation(npz_dir_list, output_dir, angle_list, axis_list, color_list,
                           smplh_fname, dmpl_fname, gif_name='sample',
                           imw=1600, imh=1600):
    """
    Produce rendered animation of input AMASS npz motion data using AMASS rendering tools. Accept
    multiple instances by putting multiple npz file directories in npz_dir_list

    Args:
        npz_dir_list (list of str): list of npz file dir for multiple instances
        output_dir (str): dir to store output images and gif
        angle_list (list of float): list of angles (one per instance, fixed)
        axis_list (list of tuple): axis of rotation for angle (must be unit vector)
        color_list (list of str): pre-defined color for each instance
                                  'pink': [0.6, 0.0, 0.4],
                                  'purple': [0.9, 0.7, 0.7],
                                  'cyan': [0.7, 0.75, 0.5],
                                  'red': [1.0, 0.0, 0.0],
                                  'green': [0.0, 1.0, 0.0],
                                  'yellow': [1.0, 1.0, 0],
                                  'brown': [0.5, 0.2, 0.1],
                                  'brown-light': [0.654, 0.396, 0.164],
                                  'blue': [0.0, 0.0, 1.0],
                                  'offwhite': [0.8, 0.9, 0.9],
                                  'white': array([1., 1., 1.]),
                                  'orange': [1.0, 0.2, 0],
                                  'grey': [0.7, 0.7, 0.7],
                                  'grey-blue': [0.345, 0.58, 0.713],
                                  'black': array([0., 0., 0.]),
                                  'yellowg': [0.83, 1, 0]}
        smplh_fname (str): body model path for smplh
        dmpls_fname (str): body model path for dmpls
        gif_name (str): name of gif
        imw (int): size of image
        imh (int): size of image
    """
    num_instances = len(npz_dir_list)
    num_betas = 16  # number of body parameters
    num_dmpls = 8  # number of DMPL parameters
    bdata = [np.load(npz_file) for npz_file in npz_dir_list]
    time_length = [len(bd['trans']) for bd in bdata]
    seq_length = min(time_length)

    body_parms = []
    comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(num_instances):
        body_parms.append({
            'root_orient': torch.Tensor(bdata[i]['poses'][:, :3]).to(comp_device),
            # controls the global root orientation
            'pose_body': torch.Tensor(bdata[i]['poses'][:, 3:66]).to(comp_device),  # controls the body
            'pose_hand': torch.Tensor(bdata[i]['poses'][:, 66:]).to(comp_device),  # controls the finger articulation
            'trans': torch.Tensor(bdata[i]['trans']).to(comp_device),  # controls the global body position
            'betas': torch.Tensor(
                np.repeat(bdata[i]['betas'][:num_betas][np.newaxis], repeats=time_length[i], axis=0)).to(
                comp_device),  # controls the body shape. Body shape is static
            'dmpls': torch.Tensor(bdata[i]['dmpls'][:, :num_dmpls]).to(comp_device)  # controls soft tissue dynamics
        })

    # Generate MeshViewer and BodyModel
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
    bm = BodyModel(bm_fname=smplh_fname, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(
        comp_device)
    faces = c2c(bm.f)

    body_pose_beta = [bm(**{k: v for k, v in body_parms[i].items() if k in ['pose_body', 'betas']})
                      for i in range(num_instances)]

    # Generate images
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(seq_length)):
        images = np.zeros([1, 1, 1, imw, imh, 3])
        body_mesh = []
        for j in range(num_instances):
            body_mesh.append(trimesh.Trimesh(vertices=c2c(body_pose_beta[j].v[i]), faces=faces,
                                             vertex_colors=np.tile(colors[color_list[j]], (6890, 1))))

            body_mesh[j].apply_transform(trimesh.transformations.rotation_matrix(
                np.radians(angle_list[j]), axis_list[j]))

        mv.set_static_meshes(body_mesh)
        images[0, 0, 0] = mv.render()

        # Save image
        img = imagearray2file(images)
        plt.imsave(output_dir + '/frame{}.png'.format(i), np.array(img)[0] / 255)

    # Generate gif
    gif_writer(output_dir, gif_name, seq_length, 'frame')