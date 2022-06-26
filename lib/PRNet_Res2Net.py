import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):                     #ARF
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = BasicConv2d(in_channel, out_channel, 1)

        self.branch10 = BasicConv2d(in_channel, out_channel, 1)
        self.branch11 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1))
        self.branch12 = BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0))
        self.branch13 = BasicConv2d(out_channel, out_channel, 3, padding=1, dilation=1)

        self.branch20 = BasicConv2d(in_channel, out_channel, 1)
        self.branch21 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2))
        self.branch22 = BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0))
        self.branch23 = BasicConv2d(out_channel, out_channel, 3, padding=2, dilation=2)

        self.branch30 = BasicConv2d(in_channel, out_channel, 1)
        self.branch31 = BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3))
        self.branch32 = BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0))
        self.branch33 = BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)

        x10 = self.branch10(x)
        x11 = self.branch11(x10)
        x12 = self.branch12(x10)
        x13 = self.branch13(x10)
        x1 = self.relu(x11 + x12 + x13)

        x20 = self.branch20(x)
        x21 = self.branch21(x20)
        x22 = self.branch22(x20)
        x23 = self.branch23(x20)
        x2 = self.relu(x21 + x22 + x23)

        x30 = self.branch30(x)
        x31 = self.branch31(x30)
        x32 = self.branch32(x30)
        x33 = self.branch33(x30)
        x3 = self.relu(x31 + x32 + x33)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4_1 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_1 = nn.Conv2d(3*channel, 1, 1)
        self.conv4_2 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5_2 = nn.Conv2d(3*channel, 1, 1)
        self.channel = channel

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x_r1 = self.conv4_1(x3_2)
        x_r1 = self.conv5_1(x_r1)

        x_r2 = torch.sigmoid(x_r1)
        x_r2 = x_r2.expand(-1, 3*self.channel, -1, -1).mul(x3_2)
        x_r2 = self.conv4_1(x_r2)
        x_r2 = self.conv5_1(x_r2)
        return x_r1, x_r2

class ReverseGuided(nn.Module):
    # res2net based encoder decoder
    def __init__(self, in_channel):
        super(ReverseGuided, self).__init__()
        self.in_channel = in_channel
        if in_channel == 2048:
            # ---- reverse attention branch 4 ----
            self.group = 8
            self.ra_conv1 = BasicConv2d(in_channel+self.group, 256, kernel_size=1)
            self.ra_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
            self.ra_conv5 = BasicConv2d(256, 1, kernel_size=1)
            self.gd_conv1 = BasicConv2d(in_channel, 256, kernel_size=5, padding=2)
            self.gd_conv2 = BasicConv2d(256, 1, kernel_size=1)
        else:
            # ---- reverse attention branch 3/2 ----
            self.group = 4
            self.ra_conv1 = BasicConv2d(in_channel+self.group, 64, kernel_size=1)
            self.ra_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
            self.ra_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
            self.gd_conv1 = BasicConv2d(in_channel, 64, kernel_size=3, padding=1)
            self.gd_conv2 = BasicConv2d(64, 1, kernel_size=1)
    def forward(self, f, crop):
        # ---- reverse attention branch_4 ----
        #crop_4 = F.interpolate(ra5_feat2, scale_factor=0.25, mode='bilinear')
        y = -1*(torch.sigmoid(crop)) + 1
        y = y.expand(-1, self.in_channel, -1, -1).mul(f)
        y = self.gd_conv1(y)
        y = self.gd_conv2(y)
        if self.group == 4:
            xs = torch.chunk(f, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(f, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        x = self.ra_conv1(x_cat)
        x = F.relu(self.ra_conv2(x))
        x = F.relu(self.ra_conv3(x))
        if self.in_channel == 2048:
            x = F.relu(self.ra_conv4(x))
            ra_feat = self.ra_conv5(x)
        else:
            ra_feat = self.ra_conv4(x)
        x = ra_feat + crop
        #lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)
        return x


class PRNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(PRNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4 = ReverseGuided(2048)
        # ---- reverse attention branch 3 ----
        self.ra3 = ReverseGuided(1024)
        # ---- reverse attention branch 2 ----
        self.ra2 = ReverseGuided(512)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        ra5_feat1, ra5_feat2 = self.agg1(x4_rfb, x3_rfb, x2_rfb)             # 留单个返回值
        lateral_map_51 = F.interpolate(ra5_feat1, scale_factor=8, mode='bilinear')
        lateral_map_52 = F.interpolate(ra5_feat2, scale_factor=8, mode='bilinear')   # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat2, scale_factor=0.25, mode='bilinear')
        x = self.ra4(x4, crop_4)
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.ra3(x3, crop_3)
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.ra2(x2, crop_2)
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return lateral_map_51, lateral_map_52, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == '__main__':
    ras = PRNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)