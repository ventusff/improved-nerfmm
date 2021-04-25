from .siren_base import SirenLayer

from collections import OrderedDict
from typing import List, Optional, Union

import torch.nn as nn
import torch
# torch.autograd.set_detect_anomaly(True)

class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super(Embedder, self).__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input: torch.Tensor):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out


def get_embedder(multires, i=0, input_dim=3):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        "include_input": True,  # needs to be True for ray_bending to work properly
        "input_dim": input_dim,
        "max_freq_log2": multires - 1,
        "N_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    return embedder_obj, embedder_obj.out_dim


class NeRF(nn.Module):
    """
    Vanilla NeRF
    """

    def __init__(
            self,
            D=8, skips=[4], W=256,
            input_ch_pts=3, input_ch_appearance=32,
            net_branch_appearance: bool = True,
            **not_used_kwargs):

        super().__init__()
        self.skips = skips
        self.input_ch_pts = input_ch_pts
        self.input_ch_appearance = input_ch_appearance
        self.net_branch_appearance = net_branch_appearance

        base_layers = []
        dim = input_ch_pts
        for i in range(D):
            base_layers.append(
                nn.Sequential(DenseLayer(dim, W, activation="relu"), nn.ReLU(inplace=True)))
            dim = W
            # skip connection after i^th layer
            if i in self.skips and i != (D-1):
                dim += input_ch_pts
        self.base_layers = nn.ModuleList(base_layers)

        if self.net_branch_appearance:
            sigma_layers = [DenseLayer(dim, 1, activation="linear"), ]
            self.sigma_layers = nn.Sequential(*sigma_layers)

            base_remap_layers = [DenseLayer(dim, W, activation="linear"), ]
            self.base_remap_layers = nn.Sequential(*base_remap_layers)

            rgb_layers = []
            dim = W + input_ch_appearance
            for i in range(1):
                rgb_layers.append(DenseLayer(dim, W // 2, activation="relu"))
                rgb_layers.append(nn.ReLU(inplace=True))
                dim = W // 2
            rgb_layers.append(DenseLayer(dim, 3, activation="linear"))
            # rgb values are normalized to [0, 1]
            rgb_layers.append(nn.Sigmoid())
            self.rgb_layers = nn.Sequential(*rgb_layers)

        else:
            output_layers = [DenseLayer(dim, 4, activation="linear"), ]
            self.output_linear = nn.Sequential(*output_layers)

    def forward(
            self,
            input_pts: torch.Tensor,
            input_views: Optional[torch.Tensor] = None):

        if input_views is None:
            shape = input_pts.shape
            shape[-1] = 0
            input_views = input_pts.new_empty(shape)

        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((input_pts, base), dim=-1)
            base = self.base_layers[i+1](base)

        if self.net_branch_appearance:
            sigma = self.sigma_layers(base)
            base_remap = self.base_remap_layers(base)
            rgb = self.rgb_layers(
                torch.cat((base_remap, input_views), dim=-1))
        else:
            outputs = self.output_linear(base)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]
            rgb = torch.sigmoid(rgb)

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret

    def query_sigma(self, input_pts: torch.Tensor):
        base = self.base_layers[0](input_pts)
        for i in range(len(self.base_layers)-1):
            if i in self.skips:
                base = torch.cat((base, base), dim=-1)
            base = self.base_layers[i+1](base)

        if self.net_branch_appearance:
            sigma = self.sigma_layers(base)
        else:
            outputs = self.output_linear(base)
            sigma = outputs[..., 3:]

        return sigma.squeeze(-1)


class SirenNeRF(nn.Module):
    """
    SIREN-ized NeRF
    """
    def __init__(
        self,
        D: int = 8, skips: List = [4], W: int = 256,
        input_ch_appearance: int = 32,
        net_branch_appearance: bool = True,
        # siren related
        sigma_mul: float = 10.,
        rgb_mul: float = 1.,
        first_layer_w0: float = 30.,
        following_layers_w0: float = 1.,
        **not_used_kwargs
    ):
        super().__init__()
        self.skips = skips
        self.net_branch_appearance = net_branch_appearance
        self.sigma_mul = sigma_mul
        self.rgb_mul = rgb_mul

        input_ch_pts = 3    # fixed. do not change.
        use_bias = True     # fixed. do not change.
        # first_layer_w0 = 30.
        # following_layers_w0 = 1.

        base_layers = [SirenLayer(input_ch_pts, W, use_bias=use_bias, w0=first_layer_w0, is_first=True)]
        for _ in range(D-1):
            base_layers.append(SirenLayer(W, W, use_bias=use_bias, w0=following_layers_w0))
        dim = W
        self.base_layers = nn.Sequential(*base_layers)

        if self.net_branch_appearance:
            sigma_layers = [nn.Linear(dim, 1, bias=use_bias), ]
            self.sigma_layers = nn.Sequential(*sigma_layers)

            base_remap_layers = [nn.Linear(dim, W, bias=use_bias), ]
            self.base_remap_layers = nn.Sequential(*base_remap_layers)

            rgb_layers = [
                SirenLayer(W + input_ch_appearance, W // 2, use_bias=use_bias, w0=following_layers_w0),
            ] + [
                nn.Linear(W // 2, 3, bias=use_bias)]
            self.rgb_layers = nn.Sequential(*rgb_layers)
        else:
            output_layers = [nn.Linear(dim, 4, bias=use_bias), ]
            self.output_linear = nn.Sequential(*output_layers)

    def forward(
            self,
            input_pts: torch.Tensor,
            input_views: Optional[torch.Tensor] = None,):
        """
        input_pts:          [(B), N, 3]
        input_views:        [(B), N, any]
        """
        if input_views is None:
            shape = list(input_pts.shape)
            shape[-1] = 0
            input_views = input_pts.new_empty(shape)

        base = input_pts
        base = self.base_layers(input_pts)

        if self.net_branch_appearance:
            sigma: torch.Tensor = self.sigma_layers(base)
            base_remap = self.base_remap_layers(base)
            rgb = torch.cat((base_remap, input_views), dim=-1)
            rgb = self.rgb_layers(rgb)
        else:
            outputs = self.output_linear(base)
            rgb = outputs[..., :3]
            sigma = outputs[..., 3:]

        # only multiply on the positive side, since a large minus sigma is meaningless.
        sigma = torch.where(sigma > 0, sigma * self.sigma_mul, sigma)
        # rgb values are normalized to [0, 1]
        rgb = torch.sigmoid_(rgb * self.rgb_mul)

        ret = OrderedDict([('rgb', rgb),
                           ('sigma', sigma.squeeze(-1))])
        return ret

    def query_sigma(self, input_pts: torch.Tensor):
        base = input_pts
        for layer in self.base_layers:
            base = layer(base)

        if self.net_branch_appearance:
            sigma = self.sigma_layers(base)
        else:
            outputs = self.output_linear(base)
            sigma = outputs[..., 3:]

        # only multiply on the positive side, since a large minus sigma is meaningless.
        sigma = torch.where(sigma > 0, sigma * self.sigma_mul, sigma)

        return sigma.squeeze(-1)


class DoubleNeRF(nn.Module):
    def __init__(self,
                 base_cls: Union[NeRF, SirenNeRF],
                 net_kwargs: dict,
                 use_fine_model=False,
                 fine_kwargs: Optional[dict] = None,
                 **kwargs):
        super().__init__()

        self.use_fine_model = use_fine_model

        self.coarse_model = base_cls(**net_kwargs, **kwargs)
        if use_fine_model:
            assert fine_kwargs is not None
            self.fine_model = base_cls(**fine_kwargs, **kwargs)

    def forward(self, *args, is_coarse: bool = True, **kwargs):
        if is_coarse:
            return self.coarse_model(*args, **kwargs)
        else:
            assert self.use_fine_model
            return self.fine_model(*args, **kwargs)

    def query_sigma(self, *args, is_coarse: bool = True, **kwargs):
        if is_coarse:
            return self.coarse_model.query_sigma(*args, **kwargs,)
        else:
            assert self.use_fine_model
            return self.fine_model.query_sigma(*args, **kwargs,)
