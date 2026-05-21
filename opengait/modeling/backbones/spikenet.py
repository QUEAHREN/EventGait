import torch
import torch.nn as nn
from copy import deepcopy

from spikingjelly.activation_based import layer, neuron, functional, surrogate
from spikingjelly import visualizing
import matplotlib.pyplot as plt
# import slayerSNN as snn




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.sn2 = spiking_neuron(**deepcopy(kwargs))
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = spiking_neuron(**deepcopy(kwargs))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.sn3(out)

        return out

class SpikingResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, spiking_neuron: callable = None, **kwargs):
        super(SpikingResNet, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(**deepcopy(kwargs))
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], spiking_neuron=spiking_neuron, **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.avgpool.step_mode == 's':
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == 'm':
            x = torch.flatten(x, 2)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


class SpikingResNet9(SpikingResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=3, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True,
                 spiking_neuron: callable = neuron.LIFNode, **kwargs):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(SpikingResNet9, self).__init__(block, layers, spiking_neuron=spiking_neuron, **kwargs)
        self.fc = None
        self.inplanes = channels[0]
        self.bn1 = layer.BatchNorm2d(self.inplanes)
        self.conv1 = layer.Conv2d(in_channel, self.inplanes, 3, 1, 1)
        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], spiking_neuron=spiking_neuron, **kwargs)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], spiking_neuron=spiking_neuron, **kwargs)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], spiking_neuron=spiking_neuron, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1, spiking_neuron=neuron.LIFNode, **kwargs):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, spiking_neuron=spiking_neuron, **kwargs)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class SimpleSNN(nn.Module):
    def __init__(self, out_channels=[32, 64, 128], strides=[2, 2, 1], v_threshold=[0.3, 1.0, 1.5]):
        super().__init__()
        self.snn = nn.Sequential(
            layer.Conv2d(3, out_channels[0], kernel_size=3, stride=strides[0], padding=1, bias=False),
            layer.BatchNorm2d(out_channels[0]),
            neuron.LIFNode(tau=2.0, v_threshold=v_threshold[0], detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=strides[1], padding=1, bias=False),
            layer.BatchNorm2d(out_channels[1]),
            neuron.LIFNode(tau=2.0, v_threshold=v_threshold[1], detach_reset=True, surrogate_function=surrogate.ATan()),
            layer.Conv2d(out_channels[1], out_channels[2], kernel_size=3, stride=strides[2], padding=1, bias=False),
            layer.BatchNorm2d(out_channels[2]),
            neuron.LIFNode(tau=2.0, v_threshold=v_threshold[2], detach_reset=True, surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, 'm')  # multi-step mode

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = x.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        x = self.snn(x)
        x = x.permute(1, 0, 2, 3, 4)  # back to batch-first (B, T, C, H, W)
        return x    # (B, 180, 128, 32, 32)


class SNNExpertLIF(nn.Module):
    def __init__(self, in_channels, out_channels, stride, tau, v_threshold=1.0, detach_reset=True, use_cupy=False):
        super().__init__()
        # separately define conv and lif to avoid the influence of weight initialization
        self.block1 = self._make_block(tau[0], in_channels, out_channels, stride, v_threshold, detach_reset)
        self.block2 = self._make_block(tau[1], in_channels, out_channels, stride, v_threshold, detach_reset)
        self.block3 = self._make_block(tau[2], in_channels, out_channels, stride, v_threshold, detach_reset)
        functional.set_step_mode(self, 'm')  # multi-step mode
        if use_cupy:
            functional.set_backend(self, 'cupy')  # use cupy backend
        self.leaky = 0.7

    def _make_block(self, tau, in_channels, out_channels, stride, v_threshold, detach_reset):
        if stride == 2:     # 下采样
            padding = 0
            kernel_size = 2
        else:
            padding = 1
            kernel_size = 3
        block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=tau, detach_reset=detach_reset, v_threshold=v_threshold)
        )
        return block
    
    def calulate_fire_rate(self, spike, info):
        # spike: (B, C, H, W, T)
        B, C, H, W, T = spike.shape
        fire_rate = torch.sum(spike) / (B * C * H * W * T)
        print('====', info, '====')
        print('Fire rate:', fire_rate.item())
    
    def print_v(self, info):
        import numpy as np
        print('---', info, '---')
        for m in self.modules():
            if isinstance(m, neuron.LIFNode):
                v = m.v
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                    print('负电位比例:', np.sum(v<0)/v.size)
                    print('正电位比例:', np.sum(v>0)/v.size)
                print('LIFNode v mean:', np.mean(v), 'max:', np.max(v), 'min:', np.min(v))


                

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        # self.print_v('before forward')      # 理论上应该全是0
        spike1 = self.block1(x)
        spike2 = self.block2(x)
        spike3 = self.block3(x)
        # self.print_v('after forward')       # 
        # 转回 batch-first，便于 MoE 处理 (B, C, H, W, T)
        spike1 = spike1.permute(1, 2, 3, 4, 0)
        spike2 = spike2.permute(1, 2, 3, 4, 0)
        spike3 = spike3.permute(1, 2, 3, 4, 0)
        # print('snn spike', self.calulate_fire_rate(spike1), self.calulate_fire_rate(spike2), self.calulate_fire_rate(spike3))
        return [spike1, spike2, spike3]   # shape = (B, C, H, W, T)


class GatingSNN(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=32, num_experts=3, 
                 tau=2.0, v_threshold=1.0):
        super().__init__()
        self.conv = nn.Sequential(
            layer.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.AdaptiveAvgPool2d((1, 1))
        )
        self.gating = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_experts)
        )
        functional.set_step_mode(self, 'm')  # multi-step mode

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.detach()
        x = x.permute(1, 0, 2, 3, 4)  # (T, B, C, H, W)
        x = self.conv(x)              # (T, B, hidden_channels, 1, 1)
        x = torch.mean(x, dim=0)    # (B, hidden_channels, 1, 1)
        x = x.view(B, -1)             # (B, hidden_channels)
        logits = self.gating(x)       # (B, num_experts)
        return torch.softmax(logits, dim=1)


class MoEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, tau, num_experts=3, v_threshold=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts = SNNExpertLIF(in_channels=in_channels, out_channels=out_channels, stride=stride, v_threshold=v_threshold, tau=tau)
        self.gating = GatingSNN(input_channels=in_channels, hidden_channels=out_channels, num_experts=num_experts)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        expert_list = self.experts(x)  # list of (B, C, H', W', T)
        expert_feats = None
        gating_weights = self.gating(x)  # (B, num_experts)

        # --- Gating Debug Save ---
        # with open('/output/gating_weights.csv', 'a') as f:
        #     for b in range(B):
        #         weights_str = ','.join([f'{w:.4f}' for w in gating_weights[b].detach().cpu().numpy()])
        #         f.write(weights_str + '\n')

        # print(f"[MoE Block] Batch Mean Weights: {gating_weights.mean(dim=0).detach().cpu().numpy().round(3)}\n")
        # --------------------------

        for i, feat in enumerate(expert_list):
            wi = gating_weights[:, i].view(B, 1, 1, 1, 1)
            expert_feats = feat * wi if expert_feats is None else expert_feats + feat * wi
        fused_feat = expert_feats.permute(0, 4, 1, 2, 3)  # (B, T, C, H', W')
        return fused_feat  # (B, T, C, H', W')
    
class MoESNN(nn.Module):
    def __init__(self, num_experts=3, out_channels=[32, 64, 128], strides=[2, 2, 1], v_threshold=[0.3, 0.7, 2.0], tau=[2.0, 3.0, 5.0]):
        super().__init__()
        self.moe = nn.Sequential(
            MoEBlock(num_experts=num_experts, in_channels=3, out_channels=out_channels[0], stride=strides[0], v_threshold=v_threshold[0], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[0], out_channels=out_channels[1], stride=strides[1], v_threshold=v_threshold[1], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[1], out_channels=out_channels[2], stride=strides[2], v_threshold=v_threshold[2], tau=tau),
        )

    def draw_heatmap(self, x, id):
        # if self.training:   
        #     return
        # x: (B, T, C, H, W)
        import shutil
        import os
        B, T, C, H, W = x.shape
        feat = x[0, :, :, :, :]  # (T, C, H, W)
        feat = feat.sum(dim=1)  # (T, H, W)
        path = f'./output/spikeheatmap{id}/'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        # visualizing.plot_2d_feature_map(feat.detach().cpu().numpy(), feat.shape[0] // 6, 6, 2, f'heatmap_sample{id}', figsize=(20, 20))
        plt.figure(figsize=(12, 12))
        for t in range(feat.shape[0]):
            heatmap = feat[t, :, :].detach().cpu().numpy()
            plt.imshow(heatmap, cmap='gray', interpolation='nearest')
            plt.axis('off')
            plt.savefig(f'{path}heatmap_sample_{t}.jpg')
            plt.close()
        print(f'Heatmaps for MoE block {id} saved.')


    def forward(self, x):
        # x: (B, T, C, H, W)
        x = self.moe[0](x)  
        # self.draw_heatmap(x, 0)
        x = self.moe[1](x)
        # self.draw_heatmap(x, 1)
        x = self.moe[2](x)
        # self.draw_heatmap(x, 2)
        return x    # (B, 180, 128, 32, 32)

class MoESNN_L(nn.Module):
    def __init__(self, num_experts=3, out_channels=[16, 32, 64, 128], strides=[1, 2, 2, 1], v_threshold=[0.3, 0.6, 1.0, 1.5], tau=[2.0, 3.0, 5.0]):
        super().__init__()
        self.moe = nn.Sequential(
            MoEBlock(num_experts=num_experts, in_channels=3, out_channels=out_channels[0], stride=strides[0], v_threshold=v_threshold[0], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[0], out_channels=out_channels[1], stride=strides[1], v_threshold=v_threshold[1], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[1], out_channels=out_channels[2], stride=strides[2], v_threshold=v_threshold[2], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[2], out_channels=out_channels[3], stride=strides[3], v_threshold=v_threshold[3], tau=tau),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = self.moe[0](x)  
        x = self.moe[1](x)
        x = self.moe[2](x)
        x = self.moe[3](x)
        return x

class MoESNN_S(nn.Module):
    def __init__(self, num_experts=3, out_channels=[32, 128], strides=[2, 2], v_threshold=[0.3, 1.0], tau=[2.0, 3.0, 5.0]):
        super().__init__()
        self.moe = nn.Sequential(
            MoEBlock(num_experts=num_experts, in_channels=3, out_channels=out_channels[0], stride=strides[0], v_threshold=v_threshold[0], tau=tau),
            MoEBlock(num_experts=num_experts, in_channels=out_channels[0], out_channels=out_channels[1], stride=strides[1], v_threshold=v_threshold[1], tau=tau),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = self.moe[0](x)  
        x = self.moe[1](x)
        return x    

class MoESNN_SS(nn.Module):
    def __init__(self, num_experts=3, out_channels=[64], strides=[2], v_threshold=[0.3], tau=[2.0, 3.0, 5.0]):
        super().__init__()
        self.moe = nn.Sequential(
            MoEBlock(num_experts=num_experts, in_channels=3, out_channels=out_channels[0], stride=strides[0], v_threshold=v_threshold[0], tau=tau),
        )
        self.pool = nn.Sequential(
            layer.Conv2d(out_channels[0], 128, kernel_size=1, stride=1, padding=0, bias=False),
            layer.MaxPool2d(3, 2, 1)
        )
        functional.set_step_mode(self, 'm')  # multi-step mode

    def forward(self, x):
        # x: (B, T, C, H, W)
        x = self.moe[0](x)  
        x = x.permute(1, 0, 2, 3, 4)  # (B, C, H, W, T)
        x = self.pool(x)              
        x = x.permute(1, 0, 2, 3, 4) 
        return x   

def gradient_check(model):
    def gradients_hook(module, grad_input, grad_output):
        print(f'Gradient check for {module}:{grad_output}')
    for name, module in model.named_modules():
        module.register_full_backward_hook(gradients_hook)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleSNN().to(device)

    x = torch.randn(1, 180, 3, 64, 64, device=device)

    out = model(x)
    print(out.shape)