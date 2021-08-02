import utils
from checkpoints import sorted_ckpts
from models.cam_params import CamParams
from models.frameworks import create_model

import os
import torch
import numpy as np

import pickle

def main_function(args):
    #--------------
    # parameters
    #--------------  

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

    export = {
        'c2w': np.ascontiguousarray(c2ws),
        'intr': np.ascontiguousarray(intr)
    }
    
    if args.out is None:
        args.out = os.path.join('.', "{}_cam.pt".format(args.expname))
    with open(args.out, 'wb') as f:
        pickle.dump(export, f)
    

if __name__ == "__main__":
    # Arguments
    parser = utils.create_args_parser()
    parser.add_argument("--out", type=str, default=None)
    args, unknown = parser.parse_known_args()
    config = utils.load_config(args, unknown)
    main_function(config)