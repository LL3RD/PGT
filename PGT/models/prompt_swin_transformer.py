import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torchvision as tv
from functools import reduce
from operator import mul
from collections import OrderedDict
from torch.nn import Conv2d, Dropout
from timm.models.layers import to_2tuple
from mmcv.runner import BaseModule, ModuleList, CheckpointLoader, load_state_dict, _load_checkpoint
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
import torch.utils.checkpoint as cp
from copy import deepcopy
from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from mmdet.models.backbones.swin import SwinTransformer, PatchEmbed
from mmdet.models.utils.transformer import AdaptivePadding
from mmdet.models.utils.ckpt_convert import swin_converter
from typing import Sequence
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

import warnings


@BACKBONES.register_module()
class PromptedSwinTransformer(BaseModule):
    def __init__(self,
                 prompt_config,
                 pretrain_img_size=224,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages

        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(PromptedSwinTransformer, self).__init__(init_cfg=init_cfg)

        # PROMPT
        self.prompt_config = prompt_config
        self.num_prompts = self.prompt_config["NUM_TOKENS"]
        self.prompt_dropout = Dropout(self.prompt_config["DROPOUT"])
        prompt_dim = self.prompt_config["PROJECT"]
        if prompt_dim > -1:
            self.prompt_proj = nn.Linear(
                prompt_dim, embed_dims)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_proj = nn.Identity()
        # PROMPT

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]
        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    num_prompts=self.num_prompts,
                    prompt_location=self.prompt_config["LOCATION"],
                    deep_prompt=self.prompt_config["DEEP"],
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                num_prompts=self.num_prompts,
                prompt_location=self.prompt_config["LOCATION"],
                deep_prompt=self.prompt_config["DEEP"],
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=int(mlp_ratio * in_channels),
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2 ** i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # PROMPT
        val = math.sqrt(6. / float(3 * reduce(mul, to_2tuple(patch_size), 1) + embed_dims))
        self.prompt_embeddings = nn.Parameter(torch.zeros(
            1, self.num_prompts, embed_dims))
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        if self.prompt_config["GPForPrompt"]:
            assert self.prompt_config["DEEP"] == False
        if self.prompt_config["DEEP"]:
            self.deep_prompt_embeddings_0 = nn.Parameter(
                torch.zeros(
                    depths[0] - 1, self.num_prompts, embed_dims
                ))
            nn.init.uniform_(
                self.deep_prompt_embeddings_0.data, -val, val)
            self.deep_prompt_embeddings_1 = nn.Parameter(
                torch.zeros(
                    depths[1], self.num_prompts, embed_dims * 2
                ))
            nn.init.uniform_(
                self.deep_prompt_embeddings_1.data, -val, val)
            self.deep_prompt_embeddings_2 = nn.Parameter(
                torch.zeros(
                    depths[2], self.num_prompts, embed_dims * 4
                ))
            nn.init.uniform_(
                self.deep_prompt_embeddings_2.data, -val, val)
            self.deep_prompt_embeddings_3 = nn.Parameter(
                torch.zeros(
                    depths[3], self.num_prompts, embed_dims * 8
                ))
            nn.init.uniform_(
                self.deep_prompt_embeddings_3.data, -val, val)

        # gradient
        for k, p in self.named_parameters():
            if "prompt" not in k:
                p.requires_grad = False
        # PROMPT

    def incorporate_prompt(self, x):
        B = x.shape[0]
        if self.prompt_config["LOCATION"] == "prepend":
            # after CLS token, all before image patches
            x = self.get_patch_embeddings(x)  # (batch_size, n_patches, hidden_dim)
            prompt_embd = self.prompt_dropout(
                self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat((
                prompt_embd, x
            ), dim=1)
        else:
            raise ValueError("Other Location is not supported")

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        # super(PromptedSwinTransformer, self).train(mode)
        self._freeze_stages()

        if mode:
            for module in self.children():
                module.train(False)
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i - 1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()

            # interpolate position bias table if needed
            relative_position_bias_table_keys = [
                k for k in state_dict.keys()
                if 'relative_position_bias_table' in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f'Error in loading {table_key}, pass')
                elif L1 != L2:
                    S1 = int(L1 ** 0.5)
                    S2 = int(L2 ** 0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(
                        nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        prompt_embd = self.prompt_dropout(
            self.prompt_embeddings.expand(B, -1, -1)
        )
        x = torch.cat(
            (prompt_embd, x), dim=1
        )

        outs = []
        if self.prompt_config["LOCATION"] == "prepend" and self.prompt_config["DEEP"]:
            for i, (stage, deep_prompt_embd) in enumerate(zip(self.stages, [
                self.deep_prompt_embeddings_0, self.deep_prompt_embeddings_1,
                self.deep_prompt_embeddings_2, self.deep_prompt_embeddings_3,
            ])):
                deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape, deep_prompt_embd)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(out)
                    # PROMPT
                    out = out[:, self.num_prompts:, :]
                    # PROMPT
                    out = out.view(-1, *out_hw_shape,
                                   self.num_features[i]).permute(0, 3, 1,
                                                                 2).contiguous()
                    outs.append(out)
        else:
            for i, stage in enumerate(self.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    out = norm_layer(out)
                    # PROMPT
                    out = out[:, self.num_prompts:, :]
                    # PROMPT
                    out = out.view(-1, *out_hw_shape,
                                   self.num_features[i]).permute(0, 3, 1,
                                                                 2).contiguous()
                    outs.append(out)
        if self.prompt_config["GPForPrompt"]:
            return outs, self.prompt_embeddings
        return outs


class PatchMerging(BaseModule):

    def __init__(self,
                 num_prompts,
                 prompt_location,
                 deep_prompt,
                 in_channels,
                 out_channels,
                 kernel_size=2,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

        # PROMPT
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if not deep_prompt:
            self.prompt_upsampling = None
            # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
        else:
            self.prompt_upsampling = None
        # PROMPT

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x, input_size):
        """
        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f'Expect ' \
                                                 f'input_size is ' \
                                                 f'`Sequence` ' \
                                                 f'but get {input_size}'

        H, W = input_size

        # PROMPT
        prompt_emb = x[:, :self.num_prompts, :]
        x = x[:, self.num_prompts:, :]
        L = L - self.num_prompts
        prompt_emb = self.upsample_prompt(prompt_emb)
        # PROMPT

        assert L == H * W, 'input feature has wrong size'

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] *
                 (self.sampler.kernel_size[0] - 1) -
                 1) // self.sampler.stride[0] + 1
        out_w = (W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] *
                 (self.sampler.kernel_size[1] - 1) -
                 1) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C

        # PROMPT
        x = torch.cat((prompt_emb, x), dim=1)
        # PROMPT

        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class SwinBlockSequence(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 # prompt
                 num_prompts=None,
                 prompt_location=None,
                 deep_prompt=None):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                num_prompts,
                prompt_location,
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None)
            self.blocks.append(block)

        self.deep_prompt = deep_prompt
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

        self.downsample = downsample

    def forward(self, x, hw_shape, deep_prompt_embd=None):
        if not self.deep_prompt:
            for block in self.blocks:
                x = block(x, hw_shape)
        else:
            B = x.shape[0]
            num_blocks = len(self.blocks)
            if deep_prompt_embd.shape[0] != num_blocks:
                for i in range(num_blocks):
                    if i == 0:
                        x = self.blocks[i](x, hw_shape)
                    else:
                        prompt_emb = deep_prompt_embd[i - 1].expand(B, -1, -1)
                        x = torch.cat(
                            (prompt_emb, x[:, self.num_prompts:, :]),
                            dim=1
                        )
                        x = self.blocks[i](x, hw_shape)
            else:
                for i in range(num_blocks):
                    prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    x = torch.cat(
                        (prompt_emb, x[:, self.num_prompts:, :]),
                        dim=1
                    )
                    x = self.blocks[i](x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class WindowMSA(BaseModule):
    def __init__(self,
                 num_prompts,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_prompts = num_prompts
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # PROMPT
        _C, _H, _W = relative_position_bias.shape
        relative_position_bias = torch.cat((
            torch.zeros(_C, self.num_prompts, _W, device=attn.device),
            relative_position_bias
        ), dim=1)
        relative_position_bias = torch.cat((
            torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
            relative_position_bias
        ), dim=-1)
        # PROMPT

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]

            # PROMPT
            mask = torch.cat((
                torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                mask), dim=1)
            mask = torch.cat((
                torch.zeros(
                    nW, _H + self.num_prompts, self.num_prompts,
                    device=attn.device),
                mask), dim=-1)
            # PROMPT

            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    def __init__(self,
                 num_prompts,
                 prompt_location,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            num_prompts=num_prompts,
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        # PROMPT
        prompt_emb = query[:, :self.num_prompts, :]
        query = query[:, self.num_prompts:, :]
        L = L - self.num_prompts
        # PROMPT
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size ** 2, C)

        # PROMPT
        num_windows = int(query_windows.shape[0] / B)
        prompt_emb = prompt_emb.unsqueeze(0)
        prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
        prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
        query_windows = torch.cat((prompt_emb, query_windows), dim=1)
        # PROMPT

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # PROMPT
        prompt_emb = attn_windows[:, :self.num_prompts, :]
        attn_windows = attn_windows[:, self.num_prompts:, :]
        prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
        prompt_emb = prompt_emb.mean(0)
        # PROMPT

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # PROMPT
        x = torch.cat((prompt_emb, x), dim=1)
        # PROMPT

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    def __init__(self,
                 num_prompts,
                 prompt_location,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            num_prompts,
            prompt_location,
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


if __name__ == '__main__':
    # print(np.random.random((2, 3)))
    # print(np.concatenate([np.random.random((2, 3)), np.array([[0, 0, 1]])], axis=0))
    #
    # xxx
    prompt_config = dict()
    prompt_config["DEEP"] = False
    prompt_config["LOCATION"] = "prepend"
    prompt_config["NUM_TOKENS"] = 5
    prompt_config["INITIATION"] = "random"
    prompt_config["DROPOUT"] = 0.0
    prompt_config["PROJECT"] = -1
    prompt_config["GPForPrompt"]=True

    checkpoint_base = "/data2/huangjunjia/coco/swin_base_patch4_window12_384_22k.pth"

    PromptedSwin = PromptedSwinTransformer(prompt_config,
                                           pretrain_img_size=384,
                                           embed_dims=128,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[4, 8, 16, 32],
                                           window_size=12,
                                           frozen_stages=-1)
    PromptedSwin.train()
    model_total_params = sum(p.numel() for p in PromptedSwin.parameters())
    model_grad_params = sum(p.numel() for p in PromptedSwin.parameters() if p.requires_grad)
    print("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    print("tuned percent:%.3f" % (model_grad_params / model_total_params * 100))

    # checkpoint = torch.load(checkpoint_base, map_location="cpu")
    # state_dict = checkpoint["model"]
    # PromptedSwin.load_state_dict(state_dict=state_dict)
    #
    input = torch.zeros((2, 3, 500, 500))
    output = PromptedSwin(input)
    # print(output)
