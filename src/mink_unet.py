# Copyright 2025 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum
from collections.abc import Sequence
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = (
        6,
        "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS")

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Convert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CUBE
}

# int_to_region_type = {m.value: m for m in ME.RegionType}
int_to_region_type = {m: ME.RegionType(m) for m in range(3)}


def convert_region_type(region_type):
    """Convert the integer region_type to the corresponding
    RegionType enum object.
    """
    return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims
        # and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def layer(
  block,
  num_blocks,
  in_channels,
  out_channels,
  dilation=1,
  conv_type=ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS,  
  bn_momentum=0.1
):
  downsample = None
  if in_channels != out_channels:
    downsample = nn.Sequential(
      conv(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=1,
          stride=1,
          bias=False,
          D=3
      ),
      ME.MinkowskiBatchNorm(out_channels, momentum=bn_momentum)
    )  
  layers = []
  layer1 = block(
    in_channels,
    out_channels,
    stride=1,
    dilation=dilation,
    downsample=downsample,
    conv_type=conv_type,
    D=3
  )
  layers.append(layer1)

  for _ in range(1, num_blocks):
    layer = block(
      out_channels,
      out_channels,
      stride=1,
      dilation=dilation,
      conv_type=conv_type,
      D=3
    )
    layers.append(layer)

  return nn.Sequential(*layers)


def conv(
  in_channels,
  out_channels,
  kernel_size,
  stride=1,
  dilation=1,
  bias=False,
  conv_type=None,
  D=3
):
  kernel_generator = ME.KernelGenerator(
    (kernel_size, kernel_size, kernel_size),
    stride,
    dilation,
    region_type=ME.RegionType.HYPER_CUBE,
    axis_types=None,  # axis_types JONAS
    dimension=D)

  return ME.MinkowskiConvolution(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    dilation=dilation,
    bias=bias,
    kernel_generator=kernel_generator,
    dimension=D)


def conv_tr(
  in_channels,
  out_channels,
  kernel_size,
  upsample_stride=1,
  dilation=1,
  bias=False,
  conv_type=ConvType.HYPERCUBE,
  D=3
):
  region_type, axis_types, kernel_size = convert_conv_type(
    conv_type, kernel_size, D)
  kernel_generator = ME.KernelGenerator(
    kernel_size,
    upsample_stride,
    dilation,
    region_type=region_type,
    axis_types=axis_types,
    dimension=D)

  return ME.MinkowskiConvolutionTranspose(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=upsample_stride,
    dilation=dilation,
    bias=bias,
    kernel_generator=kernel_generator,
    dimension=D)


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_type=ConvType.HYPERCUBE,
                 bn_momentum=0.1,
                 D=3):
        super().__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D)
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D)
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM