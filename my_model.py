import torch
import torch.nn as nn
from resnet import resnet50
import torch.nn.functional as F


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),

            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


"""Selective Feature Decoupler"""


class Mine(nn.Module):
    """
    definition of MINE
    """

    def __init__(self, input_size, hidden_size=200):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.fc2(output)
        output = F.relu(output)
        output = self.fc3(output)
        return output


class selective_feature_decoupler(nn.Module):
    """
    definition of SFD
    """

    def __init__(self, in_c, out_c, in_h, in_w):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w

        # 3*3CBR 2
        self.c1 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )
        # 3*3CBR 2
        self.c2 = nn.Sequential(
            CBR(in_c, in_c, kernel_size=3, padding=1),
            CBR(in_c, out_c, kernel_size=3, padding=1)
        )

        # before MINE, reduce channel and size
        self.reduce_c = nn.Sequential(
            CBR(out_c, 16, kernel_size=1, padding=0),
            nn.MaxPool2d(2, stride=2)
        )

        # init MINE
        self.mine_net = Mine(input_size=16 * (in_h // 2) * (in_w // 2) * 2)

    def mutual_information(self, joint, marginal):
        joint = joint.float().cuda() if torch.cuda.is_available() else joint.float()
        marginal = marginal.float().cuda() if torch.cuda.is_available() else marginal.float()

        t = self.mine_net(joint)
        et = torch.exp(self.mine_net(marginal))
        mi_lb = torch.mean(t) - torch.log(1 + torch.mean(et))

        return mi_lb

    def forward(self, x):
        # x:[B,C,H,W]
        s = self.c1(x)  # significant feature
        u = self.c2(x)  # unimportant feature

        # reduce channel
        s_16 = self.reduce_c(s)
        u_16 = self.reduce_c(u)

        # flatten s and u
        s_flat = torch.flatten(s_16, start_dim=1)
        u_flat = torch.flatten(u_16, start_dim=1)

        # create joint and marginal
        joint = torch.cat([s_flat, u_flat], dim=1)
        marginal = torch.cat([s_flat, torch.roll(u_flat, shifts=1, dims=0)], dim=1)

        # calculate mi loss
        mi_lb = self.mutual_information(joint, marginal)
        loss_mi = mi_lb

        # sigmoid
        loss_mi = torch.sigmoid(loss_mi)

        return s, loss_mi


"""Edge Mapping Engine"""


class edge_map_engine(nn.Module):
    def __init__(self, in_c=64):
        super().__init__()
        self.down1 = nn.Sequential(
            CBR(in_c, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2)
        )

        self.down2 = nn.Sequential(
            CBR(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),  # MaxPooling
            CBR(256, 128, kernel_size=3, padding=1)
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 1*1 cbr lateral
        self.c_cat = CBR(128, 128, kernel_size=1, padding=0)
        # cbr fusion
        self.up_conv1 = CBR(256, 128, kernel_size=3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 1*1 cbr lateral
        self.c_cat2 = CBR(64, 128, kernel_size=1, padding=0)
        # cbr fusion
        self.up_conv2 = CBR(256, 128, kernel_size=3, padding=1)

        # edge pred.
        self.pred_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        # bottom-up
        x1 = self.down1(x)
        x2 = self.down2(x1)

        # top-down and fusion
        x_up = self.up1(x2)
        x1_cat = self.c_cat(x1)
        x_up = torch.cat([x1_cat, x_up], dim=1)
        x_up = self.up_conv1(x_up)
        x_up = self.up2(x_up)
        x_cat = self.c_cat2(x)
        x_up = torch.cat([x_cat, x_up], dim=1)
        x_last = self.up_conv2(x_up)

        # edge pred.
        x_pred = self.pred_block(x_last)
        return x_pred, x_last


"""Edge Information Injector"""


class edge_information_injector(nn.Module):
    def __init__(self, in_c):
        super(edge_information_injector, self).__init__()

        self.conv = nn.Conv2d(128, in_c, kernel_size=1)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_c, in_c // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_c // 8, in_c, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_attention_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_attention_sigmoid = nn.Sigmoid()

        self.output_conv = nn.Conv2d(128 + in_c, in_c, kernel_size=1)

    def forward(self, f_s, f_e):
        _, _, H, W = f_s.size()
        f_e = F.adaptive_avg_pool2d(f_e, (H, W))
        f_s_conv = self.conv(f_s)
        f_e_conv = self.conv(f_e)

        ca = self.channel_attention(f_e_conv)
        f_e_att = ca * f_e_conv

        avg_out = torch.mean(f_e_att, dim=1, keepdim=True)
        max_out, _ = torch.max(f_e_att, dim=1, keepdim=True)
        spatial_in = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.spatial_attention_conv(spatial_in)
        spatial_attention = self.spatial_attention_sigmoid(spatial_attention)
        f_e_att = f_e_att * spatial_attention

        # fuse
        fused = torch.cat([f_s_conv, f_e_att], dim=1)
        output = self.output_conv(fused)

        return output


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, scale=2):  # in_c: 128, out_c: 128, scale: 2
        super().__init__()
        self.scale = scale
        self.relu = nn.ReLU(inplace=True)

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.c1 = CBR(in_c + out_c, out_c, kernel_size=1, padding=0)
        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)
        self.c4 = CBR(out_c, out_c, kernel_size=1, padding=0, act=False)
        self.ca = channel_attention(out_c)
        self.sa = spatial_attention()

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], axis=1)
        x = self.c1(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        s3 = x
        x = self.c4(x)
        x = self.relu(x + s3 + s2 + s1)

        x = self.ca(x)
        x = self.sa(x)
        return x


class output_block(nn.Module):
    """
    Output Block
    """

    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.up(x)
        x = self.c1(x)
        return x


class feature_fusion_module(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.up_2x2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_4x4 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)

        self.c11 = CBR(in_c[0], out_c, kernel_size=1, padding=0)
        self.c12 = CBR(in_c[1], out_c, kernel_size=1, padding=0)
        self.c13 = CBR(in_c[2], out_c, kernel_size=1, padding=0)
        self.c14 = CBR(out_c * 3, out_c, kernel_size=1, padding=0)

        self.c2 = CBR(out_c, out_c, act=False)
        self.c3 = CBR(out_c, out_c, act=False)

    def forward(self, x1, x2, x3):
        x1 = self.up_4x4(x1)
        x2 = self.up_2x2(x2)

        x1 = self.c11(x1)
        x2 = self.c12(x2)
        x3 = self.c13(x3)
        x = torch.cat([x1, x2, x3], axis=1)
        x = self.c14(x)

        s1 = x
        x = self.c2(x)
        x = self.relu(x + s1)

        s2 = x
        x = self.c3(x)
        x = self.relu(x + s2 + s1)

        return x


class EPSS(nn.Module):
    def __init__(self, H=256, W=256):
        super().__init__()

        self.H = H
        self.W = W

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        """ EME """
        self.eme = edge_map_engine(64)

        """SFD"""
        self.s1 = selective_feature_decoupler(64, 128, H // 2, W // 2)
        self.s2 = selective_feature_decoupler(256, 128, H // 4, W // 4)
        self.s3 = selective_feature_decoupler(512, 128, H // 8, W // 8)
        self.s4 = selective_feature_decoupler(1024, 128, H // 16, W // 16)

        """ Decoder """
        self.d1 = decoder_block(128, 128, scale=2)
        self.a1 = edge_information_injector(128)

        self.d2 = decoder_block(128, 128, scale=2)
        self.a2 = edge_information_injector(128)

        self.d3 = decoder_block(128, 128, scale=2)
        self.a3 = edge_information_injector(128)

        """ FFM """
        self.ag = feature_fusion_module([128, 128, 128], 128)

        """ Output Block"""
        self.y1 = output_block(128, 1)

    def forward(self, image):
        """ Backbone: ResNet50 """
        x0 = image
        x1 = self.layer0(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)

        """ EME """
        edge_pred, edge_info = self.eme(x1)

        """ SFD """
        s1, loss_mi1 = self.s1(x1)
        s2, loss_mi2 = self.s2(x2)
        s3, loss_mi3 = self.s3(x3)
        s4, loss_mi4 = self.s4(x4)

        """ Decoder """
        d1 = self.d1(s4, s3)
        f1 = self.a1(d1, edge_info)

        d2 = self.d2(f1, s2)
        f2 = self.a2(d2, edge_info)

        d3 = self.d3(f2, s1)
        f3 = self.a3(d3, edge_info)

        """ FFM """
        ag = self.ag(f1, f2, f3)

        """ Output """
        y1 = self.y1(ag)

        return y1, edge_pred, loss_mi1, loss_mi2, loss_mi3, loss_mi4


def prepare_input(res):
    x1 = torch.FloatTensor(1, 3, 256, 256).cuda()
    x2 = torch.FloatTensor(1, 5, 300).cuda()
    return dict(x=[x1, x2])
