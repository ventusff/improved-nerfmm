import utils
from checkpoints import sorted_ckpts
from vis.vis_utils import render_full
from models.cam_params import CamParams
from models.frameworks import create_model
from geometry import c2w_track_spiral, poses_avg

import os
import torch
import imageio
import numpy as np
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


def spiral_render(args):
    #--------------
    # parameters
    #--------------  
    utils.cond_mkdir('out')

    #--------------
    # Load model
    #--------------
    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])
    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    model, render_kwargs_train, render_kwargs_test, grad_vars = create_model(
        args, model_type=args.model.framework)

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


    # calculate params for generate path
    near = args.data.near
    far = args.data.far
    c2w_center = poses_avg(c2ws)
    up = c2ws[:, :3, 1].sum(0)
    rads = np.percentile(np.abs(c2ws[:, :3, 3]), 80, 0)
    focus_distance = (far - near) * 0.7 + near

    # calculate spiral path
    render_c2ws = c2w_track_spiral(c2w_center, up, rads, focus_distance, zrate=0.5, rots=2, N=100)
    render_c2ws = np.stack(render_c2ws, 0)

    rgbs, depths = render_full(intr, render_c2ws, H, W, near, far, render_kwargs_test, model, device, batch_size=4)
    imageio.mimwrite(os.path.join('out', '{}_spiral_rgb_{}x{}.mp4'.format(args.expname, H, W)), rgbs, fps=30, quality=8)
    imageio.mimwrite(os.path.join('out', '{}_spiral_depth_{}x{}.mp4'.format(args.expname, H, W)), depths, fps=30, quality=8)
    pass


def interpolate_render(args):
    #--------------
    # parameters
    #--------------  
    utils.cond_mkdir('out')
    num_views = 120

    #--------------
    # Load model
    #--------------
    device_ids = args.device_ids
    device = "cuda:{}".format(device_ids[0])
    exp_dir = args.training.exp_dir
    print("=> Experiments dir: {}".format(exp_dir))

    model, render_kwargs_train, render_kwargs_test, grad_vars = create_model(
        args, model_type=args.model.framework)

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


    # calculate params for generate path
    near = args.data.near
    far = args.data.far

    # calculate interpolate path
    key_rots = R.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = np.stack(render_c2ws, axis=0)

    rgbs, depths = render_full(intr, render_c2ws, H, W, near, far, render_kwargs_test, model, device, batch_size=4)
    imageio.mimwrite(os.path.join('out', '{}_spiral_rgb_{}x{}.mp4'.format(args.expname, H, W)), rgbs, fps=30, quality=8)
    imageio.mimwrite(os.path.join('out', '{}_spiral_depth_{}x{}.mp4'.format(args.expname, H, W)), depths, fps=30, quality=8)


if __name__ == "__main__":
    # Arguments
    parser = utils.create_args_parser()
    # custom configs of this script
    parser.add_argument('--render_type', type=str, default='spiral, interpolate', help='choices from [spiral, interpolate]')
    args, unknown = parser.parse_known_args()
    config = utils.load_config(args, unknown)
    if args.render_type == 'spiral':
        spiral_render(config)
    elif args.render_type == 'interpolate':
        interpolate_render(config)