# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None # 오류 검출 코드
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0] # 보간
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    """
    init을 보면 name, train_backbone(bool값), return_interm_layers, dilation 으로 구성된다.
    아래 build_backbone 코드를 보면 
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation) 로 되어있다.

    name :  backbone network를 결정한다. args.backbone 값이 들어간다. main.py에서 args.backbone을 보면 default가 resnet50으로 되어있는 것을 볼 수 있다.
    train_backbone, return_interm_layers : 아래 build backbone에서 결정된다.
    args.dilation : dilation 또는 dilated convolution 이라고 불리는 convolution 기법이 있다. 그것을 결정하는 부분
    """
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # resnet18,34 일 경우 out 채널은 512, 50 101 일 경우 2048이다. torch의 resnet 참고
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # return_interm layers 는 중간 과정을 출력해줄 것인지를 결정하는 bool값이다.
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

# backbone 과 position_embedding을 합쳐서 최종 backbone 모델을 완성한다.

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = [] # out : 중간 레이어들의 출력을 저장하는 리스트 
        pos = [] # pos : positional encoding 값을 담고 있다.
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype)) 

        return out, pos


def build_backbone(args):
    ## postion_encoding.py 로
    position_embedding = build_position_encoding(args)
    ## args 값 초기화 된 내용에 따라 train_backbone,  return_interm_layers 설정
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    ## BACKBONE : backbone, 위에서 설정한 train_backbone, return_interm_layers, 과 args dilation 초기값
    # 순서 : backbonebase -> backbone ->  Joiner -> Return
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 위에서 만들어진 backbone과 position_embedding 을 결합해서 최종 bacbkone model를 완성하고 return
    model = Joiner(backbone, position_embedding)

    model.num_channels = backbone.num_channels
    return model
