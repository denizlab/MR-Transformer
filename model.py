import torch
import torch.nn as nn
import numpy as np
import scipy
from torch.utils.checkpoint import checkpoint



# The function is adapted from the Vision Transformer repository:
# https://github.com/google-research/vision_transformer
# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def interpolate_posembed(posemb, num_tokens: int, has_class_token: bool):
    
    """Interpolate given positional embedding parameters into a new shape.

      Args:
        posemb: positional embedding parameters.
        num_tokens: desired number of tokens.
        has_class_token: True if the positional embedding parameters contain a
          class token.

      Returns:
        Positional embedding parameters interpolated into the new shape.
    """
    assert posemb.shape[0] == 1
    if has_class_token:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        num_tokens -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(num_tokens))
    assert gs_old ** 2 == len(posemb_grid), f'{gs_old ** 2} != {len(posemb_grid)}'
    assert gs_new ** 2 == num_tokens, f'{gs_new ** 2} != {num_tokens}'
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    return np.array(np.concatenate([posemb_tok, posemb_grid], axis=1))



class MR_Transformer(nn.Module):
    def __init__(self,
                 base_model = 'tiny',
                 num_mr_slice = 36,
                 mr_slice_size = 384,
                 patch_size = 16,
                 num_classes = 2,
                 use_checkpoint = False):
        super().__init__()
        
        mri_size = [num_mr_slice, mr_slice_size, mr_slice_size]
        self.use_checkpoint = use_checkpoint
        assert mri_size[1]%patch_size == 0
        assert mri_size[2]%patch_size == 0
        
        deit_name = 'deit_' + base_model + '_patch16_224'
        base_model = torch.hub.load('facebookresearch/deit:main', deit_name, pretrained=True)
        self.patch_embed = base_model.patch_embed.proj
        self.cls_token = base_model.cls_token
        self.blocks = base_model.blocks
        self.norm = base_model.norm
        
        emb_size = base_model.head.in_features
        self.head = nn.Linear(emb_size, num_classes)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)
        
        num_tokens = num_mr_slice * (mri_size[1]//patch_size) * (mri_size[2]//patch_size) + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, emb_size), requires_grad=True)
        
        # interpolate deit pre-trained pos embed 
        deit_pos_embed = base_model.pos_embed.detach().numpy()
        num_slice_tokens = (mri_size[1]//patch_size) * (mri_size[2]//patch_size) + 1
        interpolated_pos_embed = interpolate_posembed(deit_pos_embed, num_tokens=num_slice_tokens, has_class_token=True)
        interpolated_pos_embed = torch.from_numpy(interpolated_pos_embed)
        
        # replicate pos embed 
        replicated_pos_embed = torch.cat((interpolated_pos_embed[:,0:1,:],
                                          interpolated_pos_embed[:,1: ,:].repeat(1, num_mr_slice, 1)), 1)
        
        self.pos_embed.data.copy_(replicated_pos_embed)
        
        
    def forward(self, x):
        B,C,Z,H,W = x.shape
        x = torch.transpose(x, 1, 2)
        x = torch.reshape(x, (B*Z,C,H,W))
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = torch.reshape(x, (B,Z*x.shape[1],x.shape[2]))
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint(blk,x) # less GPU memory required but longer runtime
        else:
            for blk in self.blocks:
                x = blk(x)
        
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x
    
    