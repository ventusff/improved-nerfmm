import utils
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d import transforms as tr3d


class CamParams(nn.Module):
    def __init__(self, phi, t, f, H0=None, W0=None, so3_repr=None, intr_repr=None):
        super().__init__()
        self.extra_attr_keys = []
        self.register_extra_attr('so3_repr', so3_repr)
        self.register_extra_attr('intr_repr', intr_repr)
        self.register_extra_attr('H0', H0)  # used to calc focal length
        self.register_extra_attr('W0', W0)  # used to calc focal length
        self.phi = nn.Parameter(phi)
        self.t = nn.Parameter(t)
        self.f = nn.Parameter(f)

    @staticmethod
    def from_config(num_imgs: int, H0: float = 1000, W0: float = 1000,
        so3_repr: str = 'axis-angle', intr_repr: str = 'square', initial_fov: float = 53.):
        # Camera parameters to optimize
        # phi, t, f
        # phi, t here is for camera2world
        if so3_repr == 'quaternion':
            phi = torch.tensor([1., 0., 0., 0.])
        elif so3_repr == 'axis-angle':
            phi = torch.tensor([0., 0., 0.])
        elif so3_repr == 'rotation6D':
            phi = torch.tensor([1., 0., 0., 0., 1., 0.])
        else:
            raise RuntimeError("Please choose representation")

        phi = phi[None, :].expand(num_imgs, -1)

        t = torch.zeros(num_imgs, 3)
        sx = 0.5 / np.tan((.5 * initial_fov * np.pi/180.))
        sy = 0.5 / np.tan((.5 * initial_fov * np.pi/180.))
        f = torch.tensor([sx, sy])

        if intr_repr == 'square':
            f = torch.sqrt(f)
        elif intr_repr == 'ratio':
            pass
        elif intr_repr == 'exp':
            f = torch.log(f)
        else:
            raise RuntimeError("Please choose intr_repr")

        m = CamParams(phi.contiguous(), t.contiguous(), f.contiguous(), H0, W0, so3_repr, intr_repr)
        return m

    @staticmethod
    def from_state_dict(state_dict):
        m = CamParams(**state_dict)
        return m

    def forward(self, indices: torch.Tensor):
        fx, fy = self.get_focal()
        return self.phi[indices], self.t[indices], fx, fy

    def get_focal(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return get_focal(self.f, self.H0, self.W0, self.intr_repr)

    def get_camera2worlds(self):
        c2ws = get_camera2world(self.phi, self.t, self.so3_repr)
        return c2ws

    def get_intrinsic(self, new_H=None, new_W=None):
        scale_x = new_W/self.W0 if new_W is not None else 1.
        scale_y = new_H/self.H0 if new_H is not None else 1.
        
        fx, fy = self.get_focal()
        intr = torch.eye(3)
        cx = self.W0 / 2.
        cy = self.H0 / 2.
        # OK with grad: with produce grad_fn=<CopySlices>
        intr[0, 0] = fx * scale_x
        intr[1, 1] = fy * scale_y
        intr[0, 2] = cx * scale_x
        intr[1, 2] = cy * scale_y
        return intr

    def get_approx_bounds(self, near: float, far: float):
        fx, fy = get_focal(self.f.data.cpu(), self.H0, self.W0, self.intr_repr)
        rays_o, rays_d, _ = get_rays(self.phi.data.cpu(), self.t.data.cpu(), fx, fy, self.H0, self.W0, -1, self.so3_repr)
        rays_e = rays_o + rays_d * (far - near)
        rays_o = rays_o.reshape(-1, 3)
        rays_e = rays_e.reshape(-1, 3)
        all_points = np.concatenate([rays_o, rays_e], axis=0)
        min_points = np.min(all_points, axis=0)
        max_points = np.max(all_points, axis=0)
        return min_points, max_points

    def register_extra_attr(self, k, v):
        self.__dict__[k] = v
        self.extra_attr_keys.append(k)

    def load_state_dict(self, state_dict, strict: bool = True):
        # Load extra non-tensor parameters
        for k in self.extra_attr_keys:
            assert k in state_dict, 'could not found key: [{}] in state_dict'.format(k)
            self.__dict__[k] = state_dict[k]
        # Notice: DO NOT deep copy. we do not want meaningless memory usage
        nn_statedict = {}
        for k, v in state_dict.items():
            if k not in self.extra_attr_keys:
                nn_statedict[k] = v
        return super().load_state_dict(nn_statedict, strict=strict)

    def state_dict(self):
        sdict = super().state_dict()
        for k in self.extra_attr_keys:
            sdict[k] = self.__dict__[k]
        return sdict


def get_focal(f, H, W, intr_repr='square') -> Tuple[torch.Tensor, torch.Tensor]:
    if intr_repr == 'square':
        f = f ** 2
    elif intr_repr == 'ratio':
        f = f
    elif intr_repr == 'exp':
        f = torch.exp(f)
    else:
        raise RuntimeError("Please choose intr_repr")
    fx, fy = f
    fx = fx * W
    fy = fy * H
    return fx, fy


def get_rotation_matrix(rot, representation='quaternion'):
    if representation == 'axis-angle':
        assert rot.shape[-1] == 3
        # pytorch3d's implementation: axis-angle -> quaternion -> rotation matrix
        rot_m = tr3d.axis_angle_to_matrix(rot)
    elif representation == 'quaternion':
        assert rot.shape[-1] == 4
        quat = F.normalize(rot)
        rot_m = tr3d.quaternion_to_matrix(quat)  # [...,3,3]
    elif representation == 'rotation6D':
        assert rot.shape[-1] == 6
        rot_m = tr3d.rotation_6d_to_matrix(rot)
    else:
        raise RuntimeError("Please choose representation.")
    return rot_m


def get_camera2world(rot, trans, representation='quaternion'):

    assert rot.shape[:-1] == trans.shape[:-1]
    prefix = rot.shape[:-1]
    rot_m = get_rotation_matrix(rot, representation)
    tmp = torch.cat((rot_m.view(*prefix, 3, 3), trans.view(*prefix, 3, 1)), dim=-1)
    extend = torch.zeros(*prefix, 1, 4).to(rot.device)
    extend[..., 0, 3] = 1.
    homo_m = torch.cat((tmp, extend), dim=-2)   # [...,4,4]

    return homo_m # [...,4,4]


def get_rays(
        rot: torch.Tensor,
        trans: torch.Tensor,
        focal_x: torch.Tensor, focal_y: torch.Tensor,
        H: int,
        W: int,
        N_rays: int = -1,
        representation='quaternion',
        center_x = None, center_y = None):
    '''
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

    :return:
    '''

    if center_x is None:
        center_x = W/2.
    if center_y is None:
        center_y = H/2.

    device = rot.device
    assert rot.shape[:-1] == trans.shape[:-1]
    prefix = rot.shape[:-1] # [...]

    # pytorch's meshgrid has indexing='ij'
    # [..., N_rays]
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t().to(device).reshape([*len(prefix)*[1], H*W]).expand([*prefix, H*W])
    j = j.t().to(device).reshape([*len(prefix)*[1], H*W]).expand([*prefix, H*W])
    if N_rays > 0:
        N_rays = min(N_rays, H*W)
        select_inds = torch.from_numpy(
            np.random.choice(H*W, size=[*prefix, N_rays], replace=False)).to(device)
        i = torch.gather(i, -1, select_inds)
        j = torch.gather(j, -1, select_inds)
    else:
        select_inds = torch.arange(H*W).to(device)

    # [..., N_rays, 3]
    dirs = torch.stack(
        [
            (i - center_x) / focal_x,
            (j - center_y) / focal_y,
            torch.ones_like(i, device=device),
        ],
        -1,
    )  # axes orientations : x right, y downwards, z positive

    # ---------
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    # ---------

    if representation == 'quaternion':
        # rot: [..., 4]
        # trans: [..., 3]
        assert rot.shape[-1] == 4
        quat = tr3d.standardize_quaternion(F.normalize(rot, dim=-1))
        rays_d = tr3d.quaternion_apply(quat[..., None, :], dirs)
        rays_o = trans[..., None, :].expand_as(rays_d)
    elif representation == 'axis-angle':
        # original paper
        # rot: [..., 3]
        # trans: [..., 3]
        assert rot.shape[-1] == 3
        ## pytorch 3d implementation: axis-angle --> quaternion -->matrix
        rot_m = tr3d.axis_angle_to_matrix(rot)  # [..., 3, 3]
        # rotation: matrix multiplication
        rays_d = torch.sum(
            # [..., N_rays, 1, 3] * [..., 1, 3, 3]
            dirs[..., None, :] * rot_m[..., None, :3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = trans[..., None, :].expand_as(rays_d)
    elif representation == 'rotation6D':
        assert rot.shape[-1] == 6
        rot_m = tr3d.rotation_6d_to_matrix(rot)
        # rotation: matrix multiplication
        # rays_d = rot_m.view(*prefix, 1, 3, 3)\
        #     .expand([*prefix, N_rays, 3, 3]).flatten(0,-3).bmm(
        #     dirs.flatten(0, -2).view([-1, 3, 1]))
        rays_d = torch.sum(
            # [..., N_rays, 1, 3] * [..., 1, 3, 3]
            dirs[..., None, :] * rot_m[..., None, :3, :3], -1
        )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        rays_o = trans[..., None, :].expand_as(rays_d)
    else:
        raise RuntimeError("please choose representation")

    # [..., N_rays, 3]
    return rays_o, rays_d, select_inds


# -----------------
# camera plotting utils
# -----------------

def about2index(about):
    if len(about) != 2:
        raise ValueError("Convention must have 2 letters.")
    if about[0] == about[1]:
        raise ValueError(f"Invalid convention {about}.")
    for letter in about:
        if letter not in ("x", "y", "z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    letter2index = {'x': 0, 'y': 1, 'z': 2}
    i0 = letter2index[about[0]]
    i1 = letter2index[about[1]]
    return i0, i1


def plot_cam_trans(cam_param: CamParams, about='xy', return_img=False):
    #--------
    # get about index
    i0, i1 = about2index(about)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = cam_param.t.data.cpu()
    # x, y, z = t.unbind(-1)
    t1, t2 = t[..., i0].numpy(), t[..., i1].numpy()
    ax.plot(t1, t2, '^-')

    if return_img:
        return utils.figure_to_image(fig)
    else:
        return fig


def plot_cam_rot(cam_param: CamParams, representation: str ='quaternion', about='xy'):

    #--------
    # get about index
    i0, i1 = about2index(about)

    #---------
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    R = cam_param.phi.data.cpu()
    rot_m = get_rotation_matrix(R, representation)
    euler = tr3d.matrix_to_euler_angles(rot_m, 'XYZ')
    # rx, ry, rz = euler.unbind(-1)
    r1, r2 = euler[..., i0].numpy(), euler[..., i1].numpy()
    ax.plot(r1, r2, '^-')
    return fig

