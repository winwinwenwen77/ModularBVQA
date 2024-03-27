import torch
import torch.nn as nn
import torchvision.models as models
from CLIP import clip



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class ViTbCLIP_SpatialTemporal_modular_dropout(torch.nn.Module):
    def __init__(self, feat_len=8, sr=True, tr=True, dropout_sp=0.2, dropout_tp=0.2):
        super(ViTbCLIP_SpatialTemporal_modular_dropout, self).__init__()
        ViT_B_16, _ = clip.load("ViT-B/16")

        clip_vit_b_pretrained_features = ViT_B_16.visual

        self.feature_extraction = clip_vit_b_pretrained_features
        self.feat_len = feat_len
        self.dropout_sp = dropout_sp
        self.dropout_tp = dropout_tp

        self.base_quality = self.base_quality_regression(512, 128, 1)
        self.spatial_rec = self.spatial_rectifier(5*256*self.feat_len, self.dropout_sp) #
        self.temporal_rec = self.temporal_rectifier((256)*self.feat_len, self.dropout_tp)  #  Fast:256  Slow:2048

        self.sr = sr
        self.tr = tr
    
    def base_quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def spatial_rectifier(self, in_channels, dropout_sp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_sp), 
        )
        return regression_block

    def temporal_rectifier(self, in_channels, dropout_tp):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Dropout(p=dropout_tp),
        )
        return regression_block

    def forward(self, x, x_3D_features, lp):
        # input dimension: batch x frames x 3 x height x width
        x_size = x.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        x = self.feature_extraction(x)
        # x = self.avgpool(x)
        x = self.base_quality(x)

        x = x.view(x_size[0],-1)
        x = torch.mean(x, dim=1).unsqueeze(1)  

        if self.sr:
            lp_size = lp.shape
            lp = lp.view(lp_size[0], -1)
            spatial = self.spatial_rec(lp)
            s_ones = torch.ones_like(x)  #
            # ax+b
            sa = torch.chunk(spatial, 2, dim=1)[0]
            sa = torch.add(sa, s_ones) #
            sb = torch.chunk(spatial, 2, dim=1)[1]
        else:
            sa = torch.ones_like(x)
            sb = torch.zeros_like(x)
        qs = torch.add(torch.mul(torch.abs(sa), x), sb).squeeze(1)

        if self.tr:
            x_3D_features_size = x_3D_features.shape
            x_3D_features = x_3D_features.view(x_3D_features_size[0], -1)
            temporal = self.temporal_rec(x_3D_features)
            t_ones = torch.ones_like(x)  #
            # ax+b
            ta = torch.chunk(temporal, 2, dim=1)[0]
            ta = torch.add(ta, t_ones) #
            tb = torch.chunk(temporal, 2, dim=1)[1]
        else:
            ta = torch.ones_like(x)
            tb = torch.zeros_like(x)
        qt = torch.add(torch.mul(torch.abs(ta), x), tb).squeeze(1)

        if self.sr and self.tr:
            modular_a = torch.sqrt(torch.abs(torch.mul(sa,ta)))
            modular_b = torch.div(torch.add(sb,tb),2)
            qst = torch.add(torch.mul(modular_a, x), modular_b).squeeze(1)
        elif self.sr:
            qst = qs
        elif self.tr:
            qst = qt
        else:
            qst = x.squeeze(1)

        return x.squeeze(1), qs, qt, qst