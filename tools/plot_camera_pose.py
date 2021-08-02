import utils
from checkpoints import sorted_ckpts
from models.cam_params import CamParams
from models.frameworks import create_model

import os
import torch
import numpy as np

"""
modified from https://github.com/opencv/opencv/blob/master/samples/python/camera_calibration_show_extrinsics.py
"""

def inverse_homogeneoux_matrix(M):
    R = M[0:3, 0:3]
    T = M[0:3, 3]
    M_inv = np.identity(4)
    M_inv[0:3, 0:3] = R.T
    M_inv[0:3, 3] = -(R.T).dot(T)

    return M_inv


def transform_to_matplotlib_frame(cMo, X, inverse=False):
    M = np.identity(4)
    M[1, 1] = 0
    M[1, 2] = 1
    M[2, 1] = -1
    M[2, 2] = 0

    if inverse:
        return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
    else:
        return M.dot(cMo.dot(X))


def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    focal = 2 / (fx + fy)
    f_scale = scale_focal * focal

    # draw image plane
    X_img_plane = np.ones((4, 5))
    X_img_plane[0:3, 0] = [-width, height, f_scale]
    X_img_plane[0:3, 1] = [width, height, f_scale]
    X_img_plane[0:3, 2] = [width, -height, f_scale]
    X_img_plane[0:3, 3] = [-width, -height, f_scale]
    X_img_plane[0:3, 4] = [-width, height, f_scale]

    # draw triangle above the image plane
    X_triangle = np.ones((4, 3))
    X_triangle[0:3, 0] = [-width, -height, f_scale]
    X_triangle[0:3, 1] = [0, -2*height, f_scale]
    X_triangle[0:3, 2] = [width, -height, f_scale]

    # draw camera
    X_center1 = np.ones((4, 2))
    X_center1[0:3, 0] = [0, 0, 0]
    X_center1[0:3, 1] = [-width, height, f_scale]

    X_center2 = np.ones((4, 2))
    X_center2[0:3, 0] = [0, 0, 0]
    X_center2[0:3, 1] = [width, height, f_scale]

    X_center3 = np.ones((4, 2))
    X_center3[0:3, 0] = [0, 0, 0]
    X_center3[0:3, 1] = [width, -height, f_scale]

    X_center4 = np.ones((4, 2))
    X_center4[0:3, 0] = [0, 0, 0]
    X_center4[0:3, 1] = [-width, -height, f_scale]

    # draw camera frame axis
    X_frame1 = np.ones((4, 2))
    X_frame1[0:3, 0] = [0, 0, 0]
    X_frame1[0:3, 1] = [f_scale/2, 0, 0]

    X_frame2 = np.ones((4, 2))
    X_frame2[0:3, 0] = [0, 0, 0]
    X_frame2[0:3, 1] = [0, f_scale/2, 0]

    X_frame3 = np.ones((4, 2))
    X_frame3[0:3, 0] = [0, 0, 0]
    X_frame3[0:3, 1] = [0, 0, f_scale/2]

    if draw_frame_axis:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
    else:
        return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]


def draw_camera(ax, camera_matrix, cam_width, cam_height, scale_focal, extrinsics, annotation=True):
    from matplotlib import cm

    min_values = np.zeros((3, 1))
    min_values = np.inf
    max_values = np.zeros((3, 1))
    max_values = -np.inf

    X_moving = create_camera_model(
        camera_matrix, cam_width, cam_height, scale_focal, False)

    cm_subsection = np.linspace(0.0, 1.0, extrinsics.shape[0])
    colors = [cm.jet(x) for x in cm_subsection]

    for idx in range(extrinsics.shape[0]):
        # R, _ = cv.Rodrigues(extrinsics[idx,0:3])
        # cMo = np.eye(4,4)
        # cMo[0:3,0:3] = R
        # cMo[0:3,3] = extrinsics[idx,3:6]
        cMo = extrinsics[idx]
        for i in range(len(X_moving)):
            X = np.zeros(X_moving[i].shape)
            for j in range(X_moving[i].shape[1]):
                X[0:4, j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4, j], True)
            ax.plot3D(X[0, :], X[1, :], X[2, :], color=colors[idx])
            min_values = np.minimum(min_values, X[0:3, :].min(1))
            max_values = np.maximum(max_values, X[0:3, :].max(1))
        # modified: add an annotation of number
        if annotation:
            X = transform_to_matplotlib_frame(cMo, X_moving[0][0:4, 0], True)
            ax.text(X[0], X[1], X[2], "{}".format(idx), color=colors[idx])

    return min_values, max_values

def main_function(args):
    #--------------
    # parameters
    #--------------  
    cam_width = 0.0064 * 5 / 2 /2
    cam_height = 0.0048 * 5 / 2 /2
    scale_focal = 5.

    #--------------
    # Load model
    #--------------
    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])
    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    model, render_kwargs_train, render_kwargs_test, grad_vars = create_model(
        args, model_type=args.model.framework)
    print("=> Nerf params: ", utils.count_trainable_parameters(model))

    if args.training.ckpt_file is None or args.training.ckpt_file == 'None':
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(exp_dir, 'ckpts'))[-1]
    else:
        ckpt_file = args.training.ckpt_file

    print("=> Loading ckpt file: {}".format(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=device)
    model_dict = state_dict['model']
    model = model.to(device)
    model.load_state_dict(model_dict)

    #--------------
    # Load camera parameters
    #--------------
    cam_params = CamParams.from_state_dict(state_dict['cam_param'])
    H = cam_params.H0
    W = cam_params.W0
    c2ws = cam_params.get_camera2worlds().data.cpu().numpy()
    intr = cam_params.get_intrinsic(H, W).data.cpu().numpy()
    extrinsics = np.linalg.inv(c2ws)

    #--------------
    # Draw cameras
    #--------------
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-variable

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect("equal")
    # ax.set_aspect("auto")

    min_values, max_values = draw_camera(ax, intr, cam_width, cam_height, scale_focal, extrinsics, False)

    X_min = min_values[0]
    X_max = max_values[0]
    Y_min = min_values[1]
    Y_max = max_values[1]
    Z_min = min_values[2]
    Z_max = max_values[2]
    max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

    mid_x = (X_max+X_min) * 0.5
    mid_y = (Y_max+Y_min) * 0.5
    mid_z = (Z_max+Z_min) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    ax.set_title('Extrinsic Parameters Visualization')

    plt.show()
    print('Done')


if __name__ == "__main__":
    # Arguments
    parser = utils.create_args_parser()
    args, unknown = parser.parse_known_args()
    config = utils.load_config(args, unknown)
    main_function(config)