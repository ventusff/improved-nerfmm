from tqdm import tqdm
from typing import Dict, Optional

import torch.nn.functional as F
import torch
# torch.autograd.set_detect_anomaly(True)

def volume_render(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    near: float,
    far: float,
    network_fn,
    network_fn_fine=None,
    batched: bool = True,
    batched_info: Optional[Dict] = None,
    # configs
    detailed_output: bool = False,
    rayschunk: int = 32*1024,
    ret_raw: bool = False,
    use_viewdirs: bool = False,
    use_fine_model: bool = False,
    N_samples: int = 0,
    N_importance: int = 0,
    show_progress: bool = False,
    **kwargs):
    """ Do volume rendering
    input:
        rays_o: [(B,) N_rays, 3]					the starting point of each ray
        rays_d: [(B,) N_rays, 3]					the direction of each ray (not normalized, z==1)
        batched: whether the 0-dim is the batch-dim (B,)
    output:
        list
        [0] rgb_map:    [(B,) N_rays, 3]            the rgb pixel value of each ray
        [1] depth_map:  [(B,) N_rays]               the depth value of each ray
        [2] dict of extra information, 
                acc_map:            [(B,) N_rays]
                disp_map:           [(B,) N_rays]
                opacity_alpha:      [(B,) N_rays, N_samples, ...]
                visibility_weights: [(B,) N_rays, N_samples, ...]
                raw.xxxx:           [(B,) N_rays, N_samples, ...]
    """
    use_fine_model = use_fine_model and network_fn_fine is not None

    if batched:
        DIM_RAYS = 1
        DIM_SAMPLES = 2
        B = rays_d.shape[0]  # batch_size
        flat_vec_shape = [B, -1, 3]
    else:
        DIM_RAYS = 0
        DIM_SAMPLES = 1
        flat_vec_shape = [-1, 3]

    rays_o = torch.reshape(rays_o, flat_vec_shape).float()
    rays_d = torch.reshape(rays_d, flat_vec_shape).float()

    # ---------------
    # Render a ray chunk
    # ---------------
    def render_rayschunk(rays_o, rays_d):
        if use_viewdirs:
            # rays_d is not normalized; (view_dir is normalized) 
            # rays_d contains information about the ratio of the length of all rays with respect to the ray on the principle point.
            viewdirs = (rays_d / torch.norm(rays_d, dim=-1, keepdim=True))
        else:
            viewdirs = None

        # ---------------
        # Sample points on the rays
        # ---------------
        z_vals, pts = ray_sample_points(
            rays_o, rays_d, near, far, N_samples, batched=batched, **kwargs)

        # ---------------
        # Query network on the sampled points
        # ---------------
        # all in shape [(B,) N_rays, N_samples, ...]
        coarse_raw = batchify_query_network(
            pts, viewdirs, network_fn,
            batched=batched,
            batched_info=batched_info,
            **kwargs)

        # ---------------
        # Importance sampling
        # ---------------
        if N_importance > 0:
            # ---------------
            # Infer the weights of coarse points and do hierarchical sampling
            with torch.no_grad():
                *_, v_weights = ray_integration(
                    coarse_raw['rgb'], coarse_raw['sigma'], z_vals, rays_d, **kwargs)
                z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])

                # [(B,) N_rays, N_samples]
                fine_z_vals = sample_pdf(
                    z_vals_mid,
                    v_weights[..., 1:-1],
                    N_importance,
                    det=(kwargs.get('perturb', 0) == 0.0))

                # [(B,) N_rays, N_samples, 3]
                fine_pts = (rays_o[..., None, :] + rays_d[..., None, :] * fine_z_vals[..., :, None])

            # ---------------
            # Qeury network on the importance sampled points
            fine_raw = batchify_query_network(
                fine_pts, viewdirs, network_fn_fine if use_fine_model else network_fn,
                batched=batched,
                batched_info=batched_info,
                detailed_output=detailed_output,
                **kwargs)

            # ---------------
            # Re-organize the raw output from near to far
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=DIM_SAMPLES)
            _, indices = torch.sort(all_z_vals, dim=DIM_SAMPLES)
            all_z_vals = torch.gather(all_z_vals, DIM_SAMPLES, indices)
            # all in shape [(B,) N_rays, N_samples+N_importance, ...]
            all_raw = {}
            for k in coarse_raw.keys():
                val = torch.cat([fine_raw[k], coarse_raw[k]], dim=DIM_SAMPLES)
                view_shape = [*indices.shape, *(len(val.shape)-len(indices.shape))*[1]]
                all_raw[k] = torch.gather(
                    val, DIM_SAMPLES, indices.view(view_shape).expand_as(val))
        else:
            all_raw = coarse_raw
            all_z_vals = z_vals

        # ---------------
        # ** The integration of volume rendering **
        # ---------------
        rgb_map, depth_map, acc_map, disp_map, opacity_alpha, visibility_weights = \
            ray_integration(
                all_raw['rgb'], all_raw['sigma'], all_z_vals, rays_d, **kwargs)

        ret = {
            'rgb_map': rgb_map,
            'depth_map': depth_map,
        }

        if detailed_output:
            ret.update(
                {
                    'acc_map': acc_map,
                    'disp_map': disp_map,
                    'opacity_alpha': opacity_alpha,
                    'visibility_weights': visibility_weights,
                }
            )

        if ret_raw:
            for k, v in all_raw.items():
                ret['raw.{}'.format(k)] = v

        return ret

    ret = {}
    for i in tqdm(range(0, rays_o.shape[DIM_RAYS], rayschunk), disable=not show_progress):
        ret_i = render_rayschunk(
            rays_o[:, i:i+rayschunk] if batched else rays_o[i:i+rayschunk],
            rays_d[:, i:i+rayschunk] if batched else rays_d[i:i+rayschunk])
        for k, v in ret_i.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    for k, v in ret.items():
        ret[k] = torch.cat(v, DIM_RAYS)

    return ret['rgb_map'], ret['depth_map'], ret


def ray_sample_points(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        N_samples: int,
        batched=True,
        lindisp=False,
        perturb: float = 0.0,
        **not_used_kwargs):
    """
    args:
        rays_o: [(B,) N_rays, 3]
        rays_d: [(B,) N_rays, 3]
    return:
        z_vals: [(B,) N_rays, N_samples]
        pts:    [(B,) N_rays, N_samples, 3]
    """
    if batched:
        B, N_rays, _ = rays_o.shape
        prefix_sh = [B, N_rays, N_samples]
    else:
        N_rays, _ = rays_o.shape
        prefix_sh = [N_rays, N_samples]

    device = rays_o.device
    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * (t_vals)
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))
    z_vals = z_vals.expand(prefix_sh)

    if perturb:
        # get intervals between samples
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    # [(B,) N_rays, N_samples, 3]
    pts = (rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None])
    return z_vals, pts


def batchify_query_network(
        pts: torch.Tensor,
        viewdirs: torch.Tensor,
        network_fn,
        batched: bool = True,
        batched_info: Optional[Dict] = None,
        # configs
        netchunk: int = 1024*1024,
        detailed_output: bool = False,
        **not_used_kwargs):

    *_, N_rays, N_samples, _ = pts.shape
    if batched:
        B = pts.shape[0]
        DIM_TO_BATCHIFY = 1
        prefix = [B, N_rays, N_samples]
        flat_vec_shape = [B, N_rays*N_samples, 3]
    else:
        DIM_TO_BATCHIFY = 0
        prefix = [N_rays, N_samples]
        flat_vec_shape = [N_rays*N_samples, 3]
    DIM_POST_SHAPE = DIM_TO_BATCHIFY + 1

    def slice_chunk(x, ind, chunk):
        if x is not None:
            return x[:, ind:ind+chunk] if batched else x[ind:ind+chunk]
        else:
            return None

    if viewdirs is not None:
        viewdirs = viewdirs[..., None, :].expand(
            pts.shape).reshape(flat_vec_shape)
    pts = pts.reshape(flat_vec_shape)

    raw_ret = {}
    for i in range(0, pts.shape[DIM_TO_BATCHIFY], netchunk):
        raw_ret_i = network_fn(
            inputs=slice_chunk(pts, i, netchunk),
            viewdirs=slice_chunk(viewdirs, i, netchunk),
            batched_info=batched_info,
            detailed_output=detailed_output
        )
        for k, v in raw_ret_i.items():
            if k not in raw_ret:
                raw_ret[k] = []
            raw_ret[k].append(v)

    # all in shape [(B,) N_rays, N_samples, ...]
    for k, v in raw_ret.items():
        v = torch.cat(v, DIM_TO_BATCHIFY)
        raw_ret[k] = v.reshape([*prefix, *v.shape[DIM_POST_SHAPE:]])
    return raw_ret


def ray_integration(
        raw_rgb: torch.Tensor,
        raw_sigma: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        # configs
        raw_noise_std: float = 1.0,
        white_bkgd=False,
        sigma_clamp_mode='relu',    #[relu, softplus]
        **not_used_kwargs):
    device = raw_rgb.device
    if sigma_clamp_mode == 'relu':
        clamp_fn = F.relu_
    elif sigma_clamp_mode == 'softplus':
        clamp_fn = F.softplus
    else:
        raise RuntimeError("wrong clamp_fn")

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat(
        [
            dists,
            # 1e10 * torch.ones(dists[..., :1].shape).to(device)
            1e2 * torch.ones(dists[..., :1].shape).to(device)   # use 1e2, as in nerf-w
        ], dim=-1)

    # rays_d is not normalized; (view_dir is normalized) 
    # rays_d contains information about the ratio of the length of all rays with respect to the ray on the principle point.
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    noise = torch.randn(raw_sigma.shape, device=device) * raw_noise_std

    opacity_alpha = 1.0 - torch.exp(-clamp_fn(raw_sigma + noise) * dists)
    opacity_alpha_shifted = torch.cat(
        [
            torch.ones([*opacity_alpha.shape[:-1], 1], device=device),
            1.0 - opacity_alpha + 1e-10,
        ], dim=-1)

    visibility_weights = opacity_alpha *\
        torch.cumprod(opacity_alpha_shifted, dim=-1)[..., :-1]

    rgb_map = torch.sum(visibility_weights[..., None] * raw_rgb, -2)
    # depth_map = torch.sum(visibility_weights * z_vals, -1)
    # NOTE: to get the correct depth map, the sum of weights must be 1!
    depth_map = torch.sum(visibility_weights / (visibility_weights.sum(-1, keepdim=True)+1e-10) * z_vals, -1)
    acc_map = torch.sum(visibility_weights, -1)
    disp_map = 1.0 / torch.max(
        1e-10 * torch.ones_like(depth_map),
        # plus 1e-10 on the denominator to avoid nan
        depth_map / (torch.sum(visibility_weights, -1) + 1e-10),)

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, disp_map, opacity_alpha, visibility_weights


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1], device=device), cdf], -1
    )  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf.detach(), u, right=False)

    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, cdf.shape[-1]-1)
    # (batch, N_importance, 2) ==> (B, batch, N_importance, 2)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [*inds_g.shape[:-1], cdf.shape[-1]]  # fix prefix shape

    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), -1, inds_g)  # fix prefix shape

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom<eps] = 1
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


if __name__ == "__main__":
    H = 10
    W = 10
    B, N_rays, N_samples, N_importance = [7, H*W, 32, 32, ]

    def dummy_network(inputs, viewdirs, batched_info, detailed_output):
        N = inputs.shape[1]
        return {
            'rgb': torch.randn(B, N, 3),
            'sigma': torch.randn(B, N),
            'detail.dummya': torch.randn(B, N, 4, 4),
            'detail.dummyb': torch.randn(B, N, 4, 2, 2),
        }

    rays_o = torch.randn(B, H, W, 3)
    rays_d = torch.randn(B, H, W, 3)
    near = 0.0
    far = 1.0
    volume_render(
        rays_o, rays_d, near, far, dummy_network,
        N_samples=N_samples, N_importance=N_importance,
        rayschunk=1000)
