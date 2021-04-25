from .nerf_base import get_embedder
from .nerf_base import NeRF, SirenNeRF, DoubleNeRF

import functools
from typing import Optional

import torch
import torch.nn as nn


class SceneModel(nn.Module):
    def __init__(
        self,
        framework: str, # [NeRF, SirenNeRF]
        fg_kwargs,
        fg_fine_kwargs: Optional[dict] = None,
        embed_multires=10,
        embed_multires_view=4,
        use_fine_model=False,
        use_viewdirs=False,
        **dummy_kwargs):
        super().__init__()

        self.use_fine_model = use_fine_model
        self.use_viewdirs = use_viewdirs
        self.framework = framework

        if framework == 'NeRF':
            base_cls = NeRF
            self.embed_fn, input_ch = get_embedder(embed_multires)
            input_ch_views = 0
            if use_viewdirs:
                self.embed_fn_view, input_ch_views = get_embedder(
                    embed_multires_view)
        elif framework == 'SirenNeRF':
            base_cls = SirenNeRF
            input_ch = 3
            input_ch_views = 3 if use_viewdirs else 0
        else:
            raise RuntimeError("Please choose framework among: [NeRF, SirenNeRF]")

        self.fg_net = DoubleNeRF(
            base_cls,
            net_kwargs=fg_kwargs,
            use_fine_model=use_fine_model,
            fine_kwargs=fg_fine_kwargs,
            input_ch_pts=input_ch,
            input_ch_appearance=input_ch_views)

    def forward(self,
                inputs: torch.Tensor,
                viewdirs: Optional[torch.Tensor],
                batched_info: dict,
                detailed_output: bool = False,
                is_coarse: bool = True):

        if self.framework == 'NeRF':
            inputs = self.embed_fn(inputs)
            if self.use_viewdirs:
                viewdirs = self.embed_fn_view(viewdirs)

        return self.fg_net(inputs, viewdirs, is_coarse=is_coarse)

    # define using get_fn and partial
    def get_coarse_fn(self):
        return functools.partial(self.forward, is_coarse=True)

    def get_fine_fn(self):
        if self.use_fine_model:
            return functools.partial(self.forward, is_coarse=False)
        else:
            return None

    def query_sigma(self, inputs: torch.Tensor):
        if self.framework == 'NeRF':
            inputs = self.embed_fn(inputs)
        return self.fg_net.query_sigma(inputs)


def create_model(
        args,
        autodecoder_variables=None,
        model_type='DITGIRAFFE'):

    grad_vars = []

    if autodecoder_variables is not None:
        for embedding in autodecoder_variables:
            grad_vars += list(embedding.parameters())

    extra_configs = {}
    if 'Siren' in model_type:
        extra_configs.update(dict(
            sigma_mul=args.model.siren_sigma_mul,
            rgb_mul=args.model.setdefault('siren_rgb_mul', 1.),
            first_layer_w0=args.model.siren_sigma_mul,
            following_layers_w0=args.model.siren_following_layers_w0))

    fg_kwargs = dict(
        D=args.model.net_d,
        W=args.model.net_w,
        skips=args.model.net_skips,
        **extra_configs)

    fg_fine_kwargs = dict(
        D=args.model.setdefault('net_d_fine', 8),
        W=args.model.setdefault('net_w_fine', 256),
        skips=args.model.setdefault('net_skips_fine', 256),
        **extra_configs)

    model_kwargs = dict(
        fg_kwargs=fg_kwargs,
        net_fine_kwargs=fg_fine_kwargs,
        use_fine_model=args.model.setdefault('use_fine_model', False),
        use_viewdirs=args.model.use_viewdirs,
        embed_multires=args.model.multires,
        embed_multires_view=args.model.multires_views,
    )

    model = SceneModel(model_type, **model_kwargs)

    grad_vars += list(
        model.parameters()
    )  # model.parameters() does not contain deform_field parameters

    ##########################

    render_kwargs_train = {
        "near": args.data.near,
        "far": args.data.far,
        "perturb": args.model.perturb,
        "N_importance": args.model.N_importance,
        "network_fn": model.get_coarse_fn(),
        "network_fine": model.get_fine_fn(),
        # "network_fn": model.coarse_forward,
        # "network_fine": model.fine_forward,
        "N_samples": args.model.N_samples,
        "use_viewdirs": args.model.use_viewdirs,
        "white_bkgd": args.data.white_bkgd,
        "raw_noise_std": args.model.raw_noise_std,
        "ret_raw": True,
        # for training, rayschunk has not effect to the GPU usage, thus should set to maximum to prevent training speed loss (multiple forward).
        "rayschunk": args.model.rayschunk,
        "sigma_clamp_mode": args.model.sigma_clamp_mode
    }
    if 'netchunk' in args.model.keys():
        render_kwargs_train['netchunk'] = args.model.netchunk

    if args.model.framework == 'NeRF' or args.model.framework == 'SirenNeRF':
        render_kwargs_train["batched"] = False
        if args.data.get('batch_size', None) is not None and args.data.batch_size > 1:
            render_kwargs_train["batched"] = True
    else:
        render_kwargs_train["batched"] = True

    # NDC only good for LLFF-style forward facing data
    # if args.data.type != 'llff' or args.data.no_ndc:
    #    print('Not ndc!')
    # render_kwargs_train["ndc"] = False ## this parameter is removed
    render_kwargs_train["lindisp"] = False

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test["perturb"] = False
    render_kwargs_test["raw_noise_std"] = 0.0
    render_kwargs_test["ret_raw"] = False
    render_kwargs_test["rayschunk"] = args.model.setdefault('val_rayschunk', args.model.rayschunk)

    return model, render_kwargs_train, render_kwargs_test, grad_vars