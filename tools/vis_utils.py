from models.volume_rendering import volume_render

import torch
import numpy as np
from tqdm import tqdm


def get_rays_opencv_np(intrinsics: np.ndarray, c2w: np.ndarray, H: int, W: int):
    '''
    ray batch sampling
        < opencv / colmap convention, standard pinhole camera >
        the camera is facing [+z] direction, x right, y downwards
                    z
                   ↗
                  /
                 /
                o------> x
                |
                |
                |
                ↓ 
                y

    :param H: image height
    :param W: image width
    :param intrinsics: [3, 3] or [4,4] intrinsic matrix 
    :param c2w: [...,4,4] or [...,3,4] camera to world extrinsic matrix
    :return:
    '''
    prefix = c2w.shape[:-2] # [...]

    # [H, W]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # [H*W]
    u = u.reshape(-1).astype(dtype=np.float32) + 0.5    # add half pixel
    v = v.reshape(-1).astype(dtype=np.float32) + 0.5
    
    # [3, H*W]
    pixels = np.stack((u, v, np.ones_like(u)), axis=0)  

    # [3, H*W]
    rays_d = np.matmul(np.linalg.inv(intrinsics[:3, :3]), pixels)
    
    # [..., 3, H*W] = [..., 3, 3] @ [1,1,...,  3, H*W], with broadcasting
    rays_d = np.matmul(c2w[..., :3, :3], rays_d.reshape([*len(prefix)*[1], 3, H*W]))
    # [..., H*W, 3]
    rays_d = np.moveaxis(rays_d, -1, -2)

    # [..., 1, 3] -> [..., H*W, 3]
    rays_o = np.tile(c2w[..., None, :3, 3], [*len(prefix)*[1], H*W, 1])

    return rays_o, rays_d


def render_full(intr: np.ndarray, c2w: np.ndarray, H, W, near, far, render_kwargs, scene_model, device="cuda", batch_size=1, imgscale=True):
    rgbs = []
    depths = []
    scene_model.to(device)

    if len(c2w.shape) == 2:
        c2w = c2w[None, ...]
    render_kwargs['batched'] = True

    def to_img(tensor):
        tensor = tensor.reshape(tensor.shape[0], H, W, -1).data.cpu().numpy()
        if imgscale:
            return (255*np.clip(tensor, 0, 1)).astype(np.uint8)
        else:
            return tensor

    def render_chunk(c2w):
        rays_o, rays_d = get_rays_opencv_np(intr, c2w, H, W)
        rays_o = torch.from_numpy(rays_o).float().to(device)
        rays_d = torch.from_numpy(rays_d).float().to(device)

        with torch.no_grad():
            rgb, depth, _ = volume_render(
                rays_o=rays_o,
                rays_d=rays_d,
                detailed_output=False,   # to return acc map and disp map
                show_progress=True,
                **render_kwargs)
        if imgscale:
            depth = (depth-near)/(far-near)
        return to_img(rgb), to_img(depth)

    for i in tqdm(range(0, c2w.shape[0], batch_size), desc="=> Rendering..."):
        rgb_i, depth_i = render_chunk(c2w[i:i+batch_size])
        rgbs += [*rgb_i]
        depths += [*depth_i]

    return rgbs, depths