import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvtv2 import pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class FPN_PVT(nn.Module):
    f"""FPN_PVT
    """
    def __init__(self, num_classes=1, input_channels=3, channel=32, **kwargs):
        super(FPN_PVT, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './model/pretrained_models/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.Translayer1 = BasicConv2d( 64, channel, 1)
        self.Translayer2 = BasicConv2d(128, channel, 1)
        self.Translayer3 = BasicConv2d(320, channel, 1)
        self.Translayer4 = BasicConv2d(512, channel, 1)

        self.smooth3 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.main_out = nn.Conv2d(channel, num_classes, 1)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    @staticmethod
    def upsample(x, ref_size):
        if len(ref_size) == 4:
            size = ref_size[-2:]
        elif len(ref_size) == 2:
            size = ref_size
        else:
            raise ValueError(f"Invalid Size: {ref_size}")
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

    def forward(self, x):

        x_size = x.size()

        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]  # Nx64x88X88
        x2 = pvt[1]  # Nx128x44x44
        x3 = pvt[2]  # Nx320x22x22
        x4 = pvt[3]  # Nx512x11x11

        x1_t = self.Translayer1(x1)
        x2_t = self.Translayer2(x2)
        x3_t = self.Translayer3(x3)
        x4_t = self.Translayer4(x4)

        x3 = self._upsample_add(x4_t, x3_t)
        x3 = F.relu(self.smooth3(x3))
        x2 = self._upsample_add(x3, x2_t)
        x2 = F.relu(self.smooth2(x2))
        x1 = self._upsample_add(x2, x1_t)
        x1 = F.relu(self.smooth1(x1))

        out = self.main_out(x1)
        out = self.upsample(out, ref_size=x_size)

        return out
