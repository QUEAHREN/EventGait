import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..modules import SeparateFCs, SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateBNNecks
from einops import rearrange
import numpy as np
import cv2
import os
from .eventgait_utils.DINOv2 import vit_small
from torchvision.utils import save_image
from torchvision.transforms.functional import resize

## ResNet9 + dynamic branch + feature fusion

class DINOv2Teacher(nn.Module):
    def __init__(self, model_cfg, msg=None):
        super().__init__()
        self.msg_mgr = msg
        model_path = model_cfg['model_path']
        num_register_tokens = model_cfg['num_register_tokens']
        self.image_size = model_cfg['image_size']
        self.feature_size = model_cfg['feature_size']
        self.patch_size = model_cfg['patch_size']
        self.msg_mgr.log_info(f'load model from: {model_path}')
        self.teacher = vit_small(logger=self.msg_mgr, num_register_tokens=num_register_tokens,img_size=518) # 518为了load预训练模型
        pretrain_dict = torch.load(model_path)
        msg = self.teacher.load_state_dict(pretrain_dict, strict=True)
        n_parameters = sum(p.numel() for p in self.teacher.parameters())
        self.msg_mgr.log_info('Missing keys: {}'.format(msg.missing_keys))
        self.msg_mgr.log_info('Unexpected keys: {}'.format(msg.unexpected_keys))
        self.msg_mgr.log_info(f"=> loaded successfully '{model_path}'")
        self.msg_mgr.log_info('DINOv2 Count: {:.5f}M'.format(n_parameters / 1e6))
        self.teacher.requires_grad_(False)
        self.teacher.eval()

    def preprocess(self, x, sz, check=128):
        # reshape (3, 128, 128) to (3, 140, 140)
        B, C, H, W = x.shape
        assert H == W == check, "Input image size must be 128x128"
        return F.interpolate(x, (sz, sz), mode='bilinear', align_corners=False)

    def forward(self, x):
        # x = (B, T, 3, H, W)
        n, s, c, h, w = x.shape
        x = rearrange(x, 'n s c h w -> (n s) c h w')
        x = self.preprocess(x, self.image_size)
        with torch.no_grad():
            outs = self.teacher(x, is_training=True)    # ['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_norm_patchtokens_mid4', 'x_prenorm', 'masks']
            outs_last4 = outs["x_norm_patchtokens_mid4"].contiguous()       # torch.Size([30, 100, 1536])
            outs_tokens = outs["x_norm_patchtokens"].contiguous()           # torch.Size([30, 100, 384])
            # self.PCA(outs_tokens)
            del outs
            outs_last4 = rearrange(outs_last4.view(n, s, self.image_size//self.patch_size, self.image_size//self.patch_size, -1), 'n s h w c -> (n s) c h w').contiguous()
            outs_last4 = self.preprocess(outs_last4, self.feature_size, check=self.image_size//self.patch_size) # [ns,c,32,32]
            outs_last4 = rearrange(outs_last4.view(n, s, -1, self.feature_size, self.feature_size), 'n s c h w -> (n s) (h w) c').contiguous()
        return outs_last4       # B*T, 32*32, 1536

class MatchingUnit(nn.Module):
    def __init__(self, chunk_size=6):
        super().__init__()
        self.chunk_size = chunk_size
        self.dynamic_channel_conv = nn.Conv2d(128*6, 128, kernel_size=1, stride=1, padding=0)
        self.dynamic_channel_conv = SetBlockWrapper(self.dynamic_channel_conv)


    def forward(self, static_feat, motion_feat):
        Tm = motion_feat.shape[1]   # motion_feat = (4, 180, 128, 16, 16)
        B, C, Ts, Hs, Ws = static_feat.shape
        motion_feat = motion_feat.permute(0, 2, 1, 3, 4)  # (4, 128, 180, 32, 32)
        assert Ts * self.chunk_size == Tm, \
            f"Temporal length mismatch: static_feat Ts={Ts}, motion_feat Tm={Tm}, chunk_size={self.chunk_size}"
        fused_static = static_feat
        motion_feat = rearrange(motion_feat.view(B, C, self.chunk_size, Ts, Hs, Ws), 'b c cs t h w -> b (c cs) t h w', cs=self.chunk_size)  # (4, 128*6, 30, 32, 32)
        fused_motion = self.dynamic_channel_conv(motion_feat)  # (4, 128, 30, 32, 32)
        return fused_static, fused_motion


class FusionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder with residual shortcut: main path 256->256->512 (BN after last conv),
        # shortcut path 256->512, then add and ReLU
        decoder_main = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder_main = SetBlockWrapper(decoder_main)


    def forward(self, static_feat, motion_feat):
        feat = torch.cat([static_feat, motion_feat], dim=1)      # 4, 256, 30, 16, 16
        out_main = self.decoder_main(feat)
        return out_main


class EventGait_L(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        # 静态分支
        self.static_branch = self.get_backbone(model_cfg['backbone_cfg']['static_branch'])
        # self.pretrained_path = model_cfg['backbone_cfg']['pretrained_path']
        self.static_branch = SetBlockWrapper(self.static_branch)  
        self.channel_reduction = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
        self.channel_reduction = SetBlockWrapper(self.channel_reduction)
        # 动态分支
        self.dynamic_branch = self.get_backbone(model_cfg['backbone_cfg']['dynamic_branch'])   
        # 特征融合
        self.matching = MatchingUnit(chunk_size=6)
        self.fusion = FusionModule()
        # 分类头
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])      # 与金字塔池化对应的 金字塔全连接
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])       # 金字塔池化
        # DINOv2 Teacher
        self.teacher_dinov2 = DINOv2Teacher(model_cfg['pretrained_dinov2'], msg=self.msg_mgr)
        self.student_trans = nn.Conv2d(512, 1536, kernel_size=1, stride=1, padding=0, bias=False)
        self.student_trans = SetBlockWrapper(self.student_trans)

    def _load_params(self):
        if os.path.isfile(self.pretrained_path):
            print(f"Loading pretrained weights from {self.pretrained_path}...")
            checkpoint = torch.load(self.pretrained_path)
            model_dict = self.static_branch.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.static_branch.load_state_dict(model_dict)
            if len(pretrained_dict) == len(model_dict):
                print(f"[SUCCESS✅] Loaded {len(pretrained_dict)}/{len(model_dict)} layers from {self.pretrained_path}")
            else:
                print(f"[WARNING⚠️] Loaded {len(pretrained_dict)}/{len(model_dict)} layers from {self.pretrained_path}. Some layers were not loaded due to size mismatch or missing keys.")
        else:
            print(f"[ERROR❌] No pretrained weights found at {self.pretrained_path}, training from scratch.")

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        frames_ipts = ipts[0]      # torch.Size([Batch, 30, 3, 64, 64]) n s c h w
        events_ipts = ipts[1]        # torch.Size([Batch, 180, 3, 64, 64]) n s c h w
        if self.training:
            rgbs_ipts = ipts[2]
        if len(frames_ipts.size()) == 4:
            frames_ipts = frames_ipts.unsqueeze(1)
        frames_ipts = rearrange(frames_ipts, 'n s c h w -> n c s h w')  # torch.Size([4, 3, 30, 64, 64])

        del ipts
        static_feat = self.static_branch(frames_ipts)       # static_feat = torch.Size([4, 512, 30, 16, 16])

        if self.training:
            with torch.no_grad():
                teacher_feat = self.teacher_dinov2(rgbs_ipts)     # B*T, 16*16, 1536
            student_feat = rearrange(self.student_trans(static_feat), 'n c s h w -> (n s) (h w) c')  # B*T, 16*16, 1536

        static_feat = self.channel_reduction(static_feat)         # static_feat = (4, 128, 30, 16, 16)
        motion_feat = self.dynamic_branch(events_ipts)      # motion_feat = (4, 180, 128, 16, 16)
        # 特征融合
        static_feat, motion_feat = self.matching(static_feat, motion_feat)      # 4, 128, 30, 16, 16
        feat = self.fusion(static_feat, motion_feat)            # 4, 512, 30, 16, 16
        feat = self.TP(feat, seqL, options={"dim": 2})[0]       # 4, 512, 16, 16
        feat = self.HPP(feat)                   # 4, 512, p
        embed_1 = self.FCs(feat)    # (B, C, P)
        embed_2, logits = self.BNNecks(embed_1)  # (B, C, P)
        fused = embed_1
        retval = {
            'training_feat': {
                'triplet': {'embeddings': fused, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
                'mse': {'student_feat': student_feat, 'teacher_feat': teacher_feat}
            },
            'visual_summary': {
                'image/x': rearrange(frames_ipts,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': fused
            }
        } if self.training else {
            'training_feat': {
                'triplet': {'embeddings': fused, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/x': rearrange(frames_ipts,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': fused
            }
        }
        return retval

