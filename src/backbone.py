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


import time
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as Ops
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from .mink_unet import (NormType, ConvType, convert_conv_type,
                        layer, conv, conv_tr, get_norm, BasicBlock)


def conv_with_groups(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    bias=False,
    conv_type=ConvType.HYPERCUBE,
    D=3,
    groups=1
  ):
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
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
        dimension=D,
        groups=groups)


class InvertedResBlock(nn.Module):
  expansion = 1
  NORM_TYPE = NormType.BATCH_NORM

  def __init__(
      self,
      in_channels,
      out_channels,
      stride=1,
      dilation=1,
      downsample=None,
      conv_type=ConvType.HYPERCUBE,
      bn_momentum=0.1,
      D=3
  ):
    super().__init__()
    expansion_ratio = 2
    hidden_dim = in_channels * expansion_ratio
    self.use_residual = (in_channels==out_channels) and (stride == 1)

    # Expansion (1x1x1 convolution)
    self.expand = nn.Sequential(
      ME.MinkowskiConvolution(
        in_channels,
        hidden_dim,
        kernel_size=1,
        stride=1,
        dimension=D,
        bias=False),
      ME.MinkowskiBatchNorm(hidden_dim),
      ME.MinkowskiReLU(inplace=True)
    ) if expansion_ratio != 1 else nn.Identity()

    # Depthwise convolution (grouped conv)
    self.depthwise = nn.Sequential(
      ME.MinkowskiConvolution(
        hidden_dim,
        hidden_dim,
        kernel_size=3,
        stride=stride,
        bias=False,
        dimension=D),
      ME.MinkowskiBatchNorm(hidden_dim),
      ME.MinkowskiReLU(inplace=True)      
    )

    # Projection (1x1x1)
    self.project = nn.Sequential(
      ME.MinkowskiConvolution(
        hidden_dim,
        out_channels,
        kernel_size=1,
        stride=1,
        dimension=D,
        bias=False
      )
    )

  def forward(self, x):
    x_identical = x
    out = self.expand(x)
    out = self.depthwise(out)
    out = self.project(out)
    if self.use_residual:
      out = out + x_identical
    return out


class TinyExpertBase(BaseModule):
  """Base class for tiny expert of Minkowski U-Net.
  """
  BLOCK = None
  INIT_DIM = 4
  DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
  LAYERS = (1, 2, 2, 2, 2, 2, 2, 1)
  OUT_PIXEL_DIST = 1
  NORM_TYPE = NormType.BATCH_NORM
  NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
  CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

  def __init__(self, in_channels, out_channels, config=None, D=3):
    super().__init__()
    self._init_network(in_channels, out_channels, D)
    self._init_weights()

  def _init_network(self, in_channels, out_channels, D):
    """
    """
    # Setup net_mtadata
    bn_momentum = 0.02
    C0, C1, C2, C3, C4, C5, C6, C7, C8, C9 = 12, 32, 64, 128, 256, 256, 128, 96, 36, 24

    # Stage 0: (N, in_channels) -> (N, C0)
    self.s0_conv = conv(
      in_channels=in_channels,
      out_channels=C0,
      kernel_size=5,
      stride=1,
    )
    self.s0_norm = ME.MinkowskiBatchNorm(C0, momentum=bn_momentum)

    # Stage 1: (1, in_channels) -> (1/2, 16) -> (1/2, )
    self.s1_conv = conv(
      in_channels=C0,
      out_channels=C1,
      kernel_size=2,
      stride=2,
    )
    self.s1_norm = ME.MinkowskiBatchNorm(C1, momentum=bn_momentum)
    self.s1_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C1,
      out_channels=C1,
      bn_momentum=bn_momentum
    )

    # Stage 2: (1/4, 32) -> (1/8, 64)
    self.s2_conv = conv(
      in_channels=C1,
      out_channels=C1,
      kernel_size=2,
      stride=2,
    )
    self.s2_norm = ME.MinkowskiBatchNorm(C1, momentum=bn_momentum)
    self.s2_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C1,
      out_channels=C2,
      bn_momentum=bn_momentum
    )

    # Stage 3: (1/8, 64) -> (1/16, 96)
    self.s3_conv = conv(
      in_channels=C2,
      out_channels=C2,
      kernel_size=2,
      stride=2,
    )
    self.s3_norm = ME.MinkowskiBatchNorm(C2, momentum=bn_momentum)
    self.s3_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C2,
      out_channels=C3
    )

    # Stage 4: (1/16, ) -> (1/32, )
    self.s4_conv = conv(
      in_channels=C3,
      out_channels=C3,
      kernel_size=2,
      stride=2,
    )
    self.s4_norm = ME.MinkowskiBatchNorm(C3, momentum=bn_momentum)
    self.s4_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C3,
      out_channels=C4
    )

    # Stage 5: 
    self.s5_conv = conv_tr(
      in_channels=C4,
      out_channels=C5,
      kernel_size=2,
      upsample_stride=2,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D      
    )
    self.s5_norm = ME.MinkowskiBatchNorm(C5, momentum=bn_momentum)

    # Stage 6:
    self.s6_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C5+C3,
      out_channels=C5
    )
    self.s6_conv = conv_tr(
      in_channels=C5,
      out_channels=C6,
      kernel_size=2,
      upsample_stride=2,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D      
    )
    self.s6_norm = ME.MinkowskiBatchNorm(C6, momentum=bn_momentum)
    self.s6_mlp = conv(
      in_channels=C5,
      out_channels=out_channels // 4,
      kernel_size=1,
      stride=1
    )
    self.s6_upsample = conv_tr(
      in_channels=out_channels // 4,
      out_channels=out_channels // 4,
      kernel_size=2,
      upsample_stride=8,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D
    )

    # Stage 7:
    self.s7_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C2+C6,
      out_channels=C6
    )
    self.s7_conv = conv_tr(
      in_channels=C6,
      out_channels=C7,
      kernel_size=2,
      upsample_stride=2,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D
    )
    self.s7_norm = ME.MinkowskiBatchNorm(C7, momentum=bn_momentum)
    self.s7_mlp = conv(
      in_channels=C6,
      out_channels=out_channels // 4,
      kernel_size=1,
      stride=1
    )
    self.s7_upsample = conv_tr(
      in_channels=out_channels // 4,
      out_channels=out_channels // 4,
      kernel_size=2,
      upsample_stride=4,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D
    )

    # Stage 8:
    self.s8_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C1+C7,
      out_channels=C7,
    )
    self.s8_conv = conv_tr(
      in_channels=C7,
      out_channels=C8,
      kernel_size=2,
      upsample_stride=2,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D     
    )
    self.s8_norm = ME.MinkowskiBatchNorm(C8, momentum=bn_momentum)
    self.s8_mlp = conv(
      in_channels=C7,
      out_channels=out_channels // 4,
      kernel_size=1,
      stride=1
    )
    self.s8_upsample = conv_tr(
      in_channels=out_channels // 4,
      out_channels=out_channels // 4,
      kernel_size=2,
      upsample_stride=2,
      bias=False,
      conv_type=self.NON_BLOCK_CONV_TYPE,
      D=D
    )

    # Stage 9:
    self.s9_resblock = layer(
      block=self.BLOCK,
      num_blocks=2,
      in_channels=C0+C8,
      out_channels=C9
    )
    self.s9_mlp = conv(
      in_channels=C9,
      out_channels=out_channels // 4,
      kernel_size=1,
      stride=1
    )

    # Projection (1x1x1)
    self.project = nn.Sequential(*[
      ME.MinkowskiConvolution(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        dimension=D,
        bias=False) for _ in range(2)]
    )

    self.relu = ME.MinkowskiReLU(inplace=True)

  def forward(self, x_BC):

    show_time = False
    if show_time:
      t1 = time.time()

    # Stage 0
    x_BC = self.s0_conv(x_BC)
    x_BC = self.s0_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_s0_BC = x_BC

    # Stage 1
    x_BC = self.s1_conv(x_BC)
    x_BC = self.s1_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_BC = self.s1_resblock(x_BC)
    x_s1_BC = x_BC

    # Stage 2
    x_BC = self.s2_conv(x_BC)
    x_BC = self.s2_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_BC = self.s2_resblock(x_BC)
    x_s2_BC = x_BC

    # Stage 3
    x_BC = self.s3_conv(x_BC)
    x_BC = self.s3_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_BC = self.s3_resblock(x_BC)
    x_s3_BC = x_BC

    # Stage 4
    x_BC = self.s4_conv(x_BC)
    x_BC = self.s4_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_BC = self.s4_resblock(x_BC)

    # Stage 5
    x_BC = self.s5_conv(x_BC)
    x_BC = self.s5_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_s5_BC = x_BC

    # Stage 6
    x_BC = Ops.cat([x_s3_BC, x_s5_BC])
    x_BC = self.s6_resblock(x_BC)
    out_s6_BC = self.s6_mlp(x_BC)
    out_s6_BC = self.s6_upsample(out_s6_BC)

    x_BC = self.s6_conv(x_BC)
    x_BC = self.s6_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_s6_BC = x_BC

    # Stage 7
    x_BC = Ops.cat([x_s2_BC, x_s6_BC])
    x_BC = self.s7_resblock(x_BC)
    out_s7_BC = self.s7_mlp(x_BC)
    out_s7_BC = self.s7_upsample(out_s7_BC)

    x_BC = self.s7_conv(x_BC)
    x_BC = self.s7_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_s7_BC = x_BC

    # Stage 8
    x_BC = Ops.cat([x_s1_BC, x_s7_BC])
    x_BC = self.s8_resblock(x_BC)
    out_s8_BC = self.s8_mlp(x_BC)
    out_s8_BC = self.s8_upsample(out_s8_BC)

    x_BC = self.s8_conv(x_BC)
    x_BC = self.s8_norm(x_BC)
    x_BC = self.relu(x_BC)
    x_s8_BC = x_BC

    # Stage 9
    x_BC = Ops.cat([x_s0_BC, x_s8_BC])
    x_BC = self.s9_resblock(x_BC)
    x_BC = self.s9_mlp(x_BC)
    out_s9_BC = x_BC


    # Feature aggregation
    out_BC = Ops.cat([out_s6_BC, out_s7_BC, out_s8_BC, out_s9_BC])
    out_BC = self.project(out_BC)

    return out_BC

  def _init_weights(self):
    for m in self.modules():
      if isinstance(m, ME.MinkowskiBatchNorm):
        nn.init.constant_(m.bn.weight, 1)
        nn.init.constant_(m.bn.bias, 0)


@MODELS.register_module()
class TinyExpert(TinyExpertBase):
  """Tiny expert for Minkowski U-Net
  """
  BLOCK = BasicBlock
  INIT_DIM = 4