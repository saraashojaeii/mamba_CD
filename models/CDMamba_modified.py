from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from einops import rearrange
from models.mamba_customer import ConvMamba, L_GF_Mamba, G_GL_Mamba


def get_dwconv_layer(
        spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        bias: bool = False
):
    depth_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=in_channels,
                             strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True, groups=in_channels)
    point_conv = Convolution(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels,
                             strides=stride, kernel_size=1, bias=bias, conv_only=True, groups=1)
    return torch.nn.Sequential(depth_conv, point_conv)

class ModifiedSRCMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, groups=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groups = groups
        self.norm = nn.LayerNorm(input_dim)

        # Grouped ConvMamba (split channels across groups)
        self.mambas = nn.ModuleList([
            ConvMamba(d_model=input_dim // groups, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v2")
            for _ in range(groups)
        ])

        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, input_dim))  # Max 32x32 tokens (safe default)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)  # [B, N, C]
        pos_embed = F.interpolate(
            self.pos_embed.transpose(1, 2).reshape(1, self.input_dim, int(self.pos_embed.shape[1] ** 0.5), -1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).reshape(1, self.input_dim, -1).transpose(1, 2)  # Shape: [1, H*W, C]
        x = x + pos_embed[:, :x.shape[1], :]

        x_norm = self.norm(x)

        # Grouped Mamba
        chunks = x_norm.chunk(self.groups, dim=-1)
        out_chunks = [m(chunk) for m, chunk in zip(self.mambas, chunks)]
        x_mamba = torch.cat(out_chunks, dim=-1)

        # Gated residual
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x_out = gate * x_mamba + (1 - gate) * x

        x_out = self.proj(x_out)
        return x_out.transpose(1, 2).reshape(B, self.output_dim, H, W)

def get_srcm_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"
):
    srcm_layer = ModifiedSRCMLayer(input_dim=in_channels, output_dim=out_channels)  # Removed conv_mode
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(srcm_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return srcm_layer

class DirectionalAGLGF(nn.Module):
    def __init__(self, dim, act='silu'):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_mamba = L_GF_Mamba(dim, bimamba_type="v2", conv_mode="orignal_dinner", act=act)

    def forward(self, x1, x2):
        # Only x1 gets updated using x2
        B, C, H, W = x1.shape
        x1_ = rearrange(x1, 'b c h w -> b (h w) c')
        x2_ = rearrange(x2, 'b c h w -> b (h w) c')

        x1_ = self.norm_q(x1_)
        x2_ = self.norm_kv(x2_)

        x1_updated = self.cross_mamba(x1_, x2_)
        x1_out = rearrange(x1_updated, 'b (h w) c -> b c h w', h=H)
        return x1_out, x2  # x2 remains unchanged

class SRCMBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            conv_mode: str = "deepwise",
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        # print(conv_mode)
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.conv2 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x


class L_GF(nn.Module):
    def __init__(self, dim, conv_mode="deepwise", resdiual=False, act="silu"):
        super(L_GF, self).__init__()

        self.fusionencoder = L_GF_Mamba(dim,bimamba_type="v2", conv_mode=conv_mode, act=act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.resdiual = resdiual
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        id1 = x1
        id2 = x2
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        queryed_x1 = self.fusionencoder(x1, x2)
        queryed_x2 = self.fusionencoder(x2, x1)
        x1 = rearrange(queryed_x1, 'b (h w) c -> b c h w', h=h)
        x2 = rearrange(queryed_x2, 'b (h w) c -> b c h w', h=h)
        if self.resdiual:
            x1 = x1 + self.skip_scale*id1
            x2 = x2 + self.skip_scale*id2
        return x1, x2

class G_GF(nn.Module):
    def __init__(self, dim, resdiual=False, act="silu"):
        super(G_GF, self).__init__()
        self.fusionencoder = G_GL_Mamba(dim,bimamba_type="v2", act=act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.skip_scale = nn.Parameter(torch.ones(1))
        self.resdiual = resdiual
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        id1 = x1
        id2 = x2
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        queryed_x1 = self.fusionencoder(x1, x2)
        queryed_x2 = self.fusionencoder(x2, x1)
        x1 = rearrange(queryed_x1, 'b (h w) c -> b c h w', h=h)
        x2 = rearrange(queryed_x2, 'b (h w) c -> b c h w', h=h)
        if self.resdiual:
            x1 = x1 + self.skip_scale*id1
            x2 = x2 + self.skip_scale*id2
        return x1, x2


class AdaptiveGate(nn.Module):
    def __init__(self, in_dim, num_expert=2):
        super().__init__()

        self.gate = nn.Linear(in_dim*2, num_expert, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_l, x_g):
        x_l = rearrange(x_l, 'b c h w -> b (h w) c')
        x_g = rearrange(x_g, 'b c h w -> b (h w) c')
        x_l = torch.mean(x_l, dim=1)
        x_g = torch.mean(x_g, dim=1)
        x_l_g = torch.cat([x_l, x_g], dim=-1)
        gate_score = self.gate(x_l_g)
        gate_score_n = self.softmax(gate_score)
        return gate_score_n

class CDMamba(nn.Module):
    """
    Multi-class Change Detection Mamba-based Model.
    - Outputs segmentation for T1, T2, and (optionally) change/transition map.
    - num_classes: number of semantic classes (not binary).
    - If use_transition_head=True, outputs per-pixel transition logits (num_classes x num_classes channels).
    """

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 16,
            in_channels: int = 1,
            num_classes: int = 7,  # <-- NEW: number of semantic classes
            use_transition_head: bool = True,  # <-- NEW: output transition/change map
            conv_mode: str = "deepwise",
            local_query_model = "orignal_dinner",
            dropout_prob: float | None = None,
            act: tuple | str = ("RELU", {"inplace": True}),
            norm: tuple | str = ("GROUP", {"num_groups": 8}),
            norm_name: str = "",
            num_groups: int = 8,
            use_conv_final: bool = True,
            blocks_down: tuple = (1, 2, 2, 4),
            blocks_up: tuple = (1, 1, 1),
            mode: str = "",
            up_mode="ResMamba",
            up_conv_mode="deepwise",
            resdiual=False,
            stage = 4,
            diff_abs="later", # "later" or "early"
            mamba_act = "silu",
            upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_transition_head = use_transition_head

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.mode = mode
        self.stage = stage
        self.up_conv_mode = up_conv_mode
        self.mamba_act = mamba_act
        self.resdiual = resdiual
        self.up_mode = up_mode
        self.diff_abs = diff_abs
        self.conv_mode = conv_mode
        self.local_query_model = local_query_model
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.channels_list = [self.init_filters, self.init_filters*2, self.init_filters*4, self.init_filters*8]
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        print(self.blocks_up)
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        print(self.norm)
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.use_conv_final = use_conv_final
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers(up_mode=self.up_mode)
        self.srcm_decoder_layers_seg_t1, self.up_samples_seg_t1 = self._make_srcm_decoder_layers(up_mode=self.up_mode)
        self.srcm_decoder_layers_seg_t2, self.up_samples_seg_t2 = self._make_srcm_decoder_layers(up_mode=self.up_mode)
        # Remove old conv_final, replaced by segmentation heads
        # self.conv_final = self._make_final_conv(out_channels)

        # --- MULTI-CLASS SEGMENTATION HEADS ---
        # Each head outputs num_classes channels
        self.seg_head_t1 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )
        self.seg_head_t2 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )
        if self.use_transition_head:
            # Output 2 channels: [no-change, change]
            self.change_head = nn.Sequential(
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters * 2),
                self.act_mod,
                get_conv_layer(self.spatial_dims, self.init_filters * 2, 2, kernel_size=1, bias=True),
            )

        self.fusion_blocks = nn.ModuleList([
            DirectionalAGLGF(self.channels_list[i], act=self.mamba_act)
            for i in range(self.stage)
        ])

        # self.l_gf1 = L_GF(self.channels_list[0], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        # self.l_gf2 = L_GF(self.channels_list[1], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        # self.l_gf3 = L_GF(self.channels_list[2], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        # self.l_gf4 = L_GF(self.channels_list[3], conv_mode=self.local_query_model, resdiual=self.resdiual, act=self.mamba_act)
        # self.l_gf = nn.Sequential(self.l_gf1, self.l_gf2, self.l_gf3, self.l_gf4)


        # self.g_gf1 = G_GF(self.channels_list[0], resdiual=self.resdiual, act=self.mamba_act)
        # self.g_gf2 = G_GF(self.channels_list[1], resdiual=self.resdiual, act=self.mamba_act)
        # self.g_gf3 = G_GF(self.channels_list[2], resdiual=self.resdiual, act=self.mamba_act)
        # self.g_gf4 = G_GF(self.channels_list[3], resdiual=self.resdiual, act=self.mamba_act)
        # self.g_gf = nn.Sequential(self.g_gf1, self.g_gf2, self.g_gf3, self.g_gf4)

        # self.ag1 = AdaptiveGate(self.channels_list[0])
        # self.ag2 = AdaptiveGate(self.channels_list[1])
        # self.ag3 = AdaptiveGate(self.channels_list[2])
        # self.ag4 = AdaptiveGate(self.channels_list[3])
        # self.ag = nn.Sequential(self.ag1, self.ag2, self.ag3, self.ag4)



        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_srcm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2, conv_mode=conv_mode)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, conv_mode=conv_mode) for _ in range(item)]
            )
            srcm_encoder_layers.append(down_layer)
        return srcm_encoder_layers

    def _make_srcm_decoder_layers(self, up_mode):
        srcm_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        if up_mode == 'SRCM':
            Block_up = SRCMBlock
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            srcm_decoder_layers.append(
                nn.Sequential(
                    *[
                        Block_up(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, conv_mode=self.up_conv_mode)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return srcm_decoder_layers, up_samples

    def _make_final_conv(self, out_channels: int):
        return nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, out_channels, kernel_size=1, bias=True),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def _decode_with_layers(self, x: torch.Tensor, down_x: list[torch.Tensor],
                             up_samples: nn.ModuleList, decoder_layers: nn.ModuleList) -> torch.Tensor:
        """Generic decoder that operates on provided up-sample and decoder layer lists."""
        for i, (up, upl) in enumerate(zip(up_samples, decoder_layers)):
            x_up = up(x)
            # Ensure spatial dimensions match for skip connection
            target_size = down_x[i + 1].shape[2:]
            if x_up.shape[2:] != target_size:
                x_up = F.interpolate(x_up, size=target_size, mode='bilinear', align_corners=False)
            x = x_up + down_x[i + 1]
            x = upl(x)
        return x

    # Backward-compatibility wrapper for change decoder path
    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        return self._decode_with_layers(x, down_x, self.up_samples, self.srcm_decoder_layers)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Returns:
            seg_logits_t1: [B, num_classes, H, W] -- segmentation logits for T1
            seg_logits_t2: [B, num_classes, H, W] -- segmentation logits for T2
            change_logits: [B, num_classes*num_classes, H, W] -- transition logits (if enabled)
        """
        b, c, h, w = x1.shape
        # Encode both images
        x1_latent, down_x1 = self.encode(x1)
        x2_latent, down_x2 = self.encode(x2)
        down_x = []
        for i in range(len(down_x1)):
            x1_i, x2_i = down_x1[i], down_x2[i]
            if self.diff_abs == "later" and self.mode == "AGLGF":
                if i < self.stage:
                    x1_i, _ = self.fusion_blocks[i](x1_i, x2_i)  # only x1 gets updated
            down_x.append(torch.abs(x1_i - x2_i))
        down_x.reverse()
        # Decode change features
        fused = self.decode(down_x[0], down_x)
        # Also decode for T1 and T2 (for segmentation)
        down_x1.reverse()
        down_x2.reverse()
        seg1 = self._decode_with_layers(x1_latent, down_x1, self.up_samples_seg_t1, self.srcm_decoder_layers_seg_t1)
        seg2 = self._decode_with_layers(x2_latent, down_x2, self.up_samples_seg_t2, self.srcm_decoder_layers_seg_t2)
        seg_logits_t1 = self.seg_head_t1(seg1)
        seg_logits_t2 = self.seg_head_t2(seg2)
        if self.use_transition_head:
            # Concatenate last features for change head
            change_logits = self.change_head(torch.cat([seg1, seg2], dim=1))
            return seg_logits_t1, seg_logits_t2, change_logits
        else:
            return seg_logits_t1, seg_logits_t2

if __name__ == "__main__":
    device = "cuda:0"
    model = CDMamba(
        spatial_dims=2,
        in_channels=3,
        num_classes=6,
        init_filters=16,
        mode="AGLGF",
        stage=4,
        conv_mode="orignal",
        local_query_model="orignal_dinner",
        mamba_act="silu",
        up_mode="SRCM",
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        resdiual=False,
        diff_abs="later"
    ).to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    seg_t1, seg_t2, change = model(x, x)
    print(seg_t1.shape, seg_t2.shape, change.shape)
