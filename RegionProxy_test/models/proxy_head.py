
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, trunc_normal_init
from mmcv.utils import to_2tuple


@HEADS.register_module()
class ProxyHead(BaseDecodeHead):

    def __init__(self, in_channels, channels, num_classes, region_res=(2, 2), norm_cfg=None, act_cfg=dict(type='ReLU'),
                 init_cfg={}, *args, **kwargs):

        super(ProxyHead, self).__init__(
            in_channels, channels, num_classes=num_classes, norm_cfg=norm_cfg,
            act_cfg=act_cfg, init_cfg=init_cfg, *args, **kwargs)

        self.region_res = to_2tuple(region_res)

        self.mlp = nn.Sequential(nn.Sequential(nn.Linear(in_channels[-1], num_classes)))

        #self.merge_mlp = nn.Linear(27,9)
        self.mlp_input = nn.Linear(5* in_channels[0], 5 * in_channels[0] )
        self.mlp1 = nn.Linear(5* in_channels[0], in_channels[0] )
        self.fc1 = nn.Linear(in_channels[0],in_channels[0])
        self.fc2 = nn.Linear(in_channels[0],5 * in_channels[0])
        self.mlp2 = nn.Linear(5* in_channels[0], in_channels[0] )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.act_layer = nn.GELU()

        self.affinity_head = nn.Sequential(
            DepthwiseSeparableConvModule(
                in_channels[0], channels, kernel_size=3, padding=1, act_cfg=act_cfg, norm_cfg=norm_cfg),
            ConvModule(
                channels, 9 * self.region_res[0] * self.region_res[1], kernel_size=1, act_cfg=None)
        )

        delattr(self, 'conv_seg')

    def init_weights(self):
        super(ProxyHead, self).init_weights()
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=.0)
        assert all(self.affinity_head[-1].conv.bias == 0)

    def forward_affinity(self, x1,x2,x3,x4,x5):
        self._device = x1.device
        B, C, H, W = x1.shape
       
        feats_s = torch.cat([x1,x2,x3,x4,x5], dim = 1)
        feats_s = feats_s.reshape(B, H, W, 5 * C) # B 3C H W
        feats_s1 = self.mlp_input(feats_s)
        feats_s2 = feats_s1.reshape(B,5*C, H, W)

        feats_s = self.mlp1(feats_s1)

        feats_z = feats_s.reshape(B, C, H, W) # B C H W
        feats_z = self.act_layer(feats_z)  # B C H W
        feats_z = self.gap(feats_z)  # B C 1 1

        feats_z = self.fc1(feats_z.squeeze())  # B C 
        feats_z = self.act_layer(feats_z) # B C
        feats_z = self.fc2(feats_z) # B 3C 

        feats_v = feats_z.view(B, 5* C, 1, 1)
        feats_v = torch.softmax(feats_v, dim = 1)

        feats_s1 = feats_s2 * feats_v
        #print(feats_s1.shape)
        #print(torch.cat([x1,x2,x3,x4,x5], dim = 1).reshape(B, H, W, 5 * C).shape)
        x1 = feats_s1 + torch.cat([x1,x2,x3,x4,x5], dim = 1)
        
        """
        # print(feats_v)
        x1 = x1 * torch.squeeze(feats_v[:,0,:,:,:],1)
        x2 = x2 * torch.squeeze(feats_v[:,1,:,:,:],1)
        x3 = x1 * torch.squeeze(feats_v[:,2,:,:,:],1)
        x4 = x2 * torch.squeeze(feats_v[:,3,:,:,:],1)
        x5 = x1 * torch.squeeze(feats_v[:,4,:,:,:],1)
        """
        


        # x1 = torch.cat([x1, x2, x3, x4, x5], dim = 1)
        #print(x1.shape)
        #x1 = x1.reshape(B, H, W, 5 * C)
        #print(x1.shape)
        x1 = x1.reshape(B, H, W, 5 * C)
        x1 = self.mlp2(x1)
        #print(x1.shape)
        x1 = x1.reshape(B,C, H, W)
        #print(x1.shape)
        # get affinity
        # get affinity
        x1 = x1.contiguous()
        
        #x2 = x2.contiguous()
        #x3 = x3.contiguous()
       
        
        affinity1 = self.affinity_head(x1) #  input : (H, W, D) == (N, D)    output :  (H, W, 9hw)
        #affinity2 = self.affinity_head(x2)
        #affinity3 = self.affinity_head(x3)

        affinity1 = affinity1.reshape(B, 9, *self.region_res, H, W)  # (B, 9, h, w, H, W)
        #affinity2 = affinity2.reshape(B, 9, *self.region_res, H, W)
        #affinity3 = affinity3.reshape(B, 9, *self.region_res, H, W)

        # handle borders
        affinity1[:, :3, :, :, 0, :] = float('-inf')  # top
        affinity1[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        affinity1[:, ::3, :, :, :, 0] = float('-inf')  # left
        affinity1[:, 2::3, :, :, :, -1] = float('-inf')  # right

        #affinity2[:, :3, :, :, 0, :] = float('-inf')  # top
        #affinity2[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        #affinity2[:, ::3, :, :, :, 0] = float('-inf')  # left
        #affinity2[:, 2::3, :, :, :, -1] = float('-inf')  # right


        #affinity3[:, :3, :, :, 0, :] = float('-inf')  # top
        #affinity3[:, -3:, :, :, -1, :] = float('-inf')  # bottom
        #affinity3[:, ::3, :, :, :, 0] = float('-inf')  # left
        #affinity3[:, 2::3, :, :, :, -1] = float('-inf')  # right


        #affinity = torch.cat((affinity1,affinity2,affinity3), dim=1)
        #print(affinity.shape, "concat")
        #affinity = affinity.reshape(B, *self.region_res, H, W, 27)
        #affinity = self.merge_mlp(affinity)
        #affinity = affinity.reshape(B, 9,*self.region_res, H, W)
        #print(affinity.shape, "before softmax")
        
        # affinity = affinity1 + affinity2 + affinity3

        
        affinity = affinity1.softmax(dim=1) # 확률 분포로 변경
        return affinity

    def forward_cls(self, x):
        self._device = x.device
        B, _, H, W = x.shape

        # get token logits
        token_logits = self.mlp(x.permute(0, 2, 3, 1).reshape(B, H * W, -1))  # (B, H * W, C)
        return token_logits

    def forward(self, inputs):
        x_1,x_2, x_3, x_4, x_5, x = self._transform_inputs(inputs)  # vit block  (B, C, H, W)
        B, _, H, W = x.shape

        affinity = self.forward_affinity(x_1,x_2, 
                                         x_3, x_4, x_5) # x_mid에 대한 (B, 9, h, w, H, W) 생성
        token_logits = self.forward_cls(x)  # vit 마지막 블록에 대해 (B, H * W, C) 생성

        # classification per pixel
        token_logits = token_logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
        token_logits = F.unfold(token_logits, kernel_size=3, padding=1).reshape(B, -1, 9, H, W)  # (B, C, 9, H, W)
        token_logits = einops.rearrange(token_logits, 'B C n H W -> B H W n C')  # (B, H, W, 9, C) = M x 9 x c

        affinity = einops.rearrange(affinity, 'B n h w H W -> B H W (h w) n')  # (B, H, W, h * w, 9) == N x M
        seg_logits = (affinity @ token_logits).reshape(B, H, W, *self.region_res, -1)  # (B, H, W, h, w, C)
        seg_logits = einops.rearrange(seg_logits, 'B H W h w C -> B C (H h) (W w)')  # (B, C, H * h, W * w)

        return seg_logits
