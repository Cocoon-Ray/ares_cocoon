# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool_mae=True, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool_mae = global_pool_mae
        if self.global_pool_mae:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool_mae:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def vis_compute_flow(self, input_tensor):
        """
        a static analytic tool, which should be run after initialization in a seperate script

        Args:
            input_tensor: (bs, ch, h, w)
        """

        import subprocess
        import tempfile
        from torch.utils.tensorboard import SummaryWriter

        temp_dir = tempfile.mkdtemp()
        with SummaryWriter(log_dir=temp_dir, comment='dae') as writer:
            subprocess.Popen(['tensorboard', f'--logdir={temp_dir}', '--port=6006'])
            writer.add_graph(self, input_tensor)
        try:
            while True:
                pass
        except:
            import shutil
            shutil.rmtree(temp_dir)

def vit_base_patch16(**kwargs):
    '''The function to create vit_base_patch16 model in MAE.'''
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    '''The function to create vit_large_patch16 model in MAE.'''
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model