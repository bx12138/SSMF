import torch.nn as nn
import torch.nn.functional as F
import torch

class Encoder(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_feat, out_channels=out_feat, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out.mean(dim=-1)



class SSMF(nn.Module):
    def __init__(self, n_feat, n_cls, proj_dim=0):
        super(SSMF, self).__init__()
        self.proj_dim = proj_dim
        self.enc_A = Encoder(n_feat, 64)
        self.enc_G = Encoder(n_feat, 64)

        # transformer module
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=256, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)

        if self.proj_dim > 0:
            self.proj_A = nn.Linear(in_features=64, out_features=proj_dim, bias=False)
            self.proj_G = nn.Linear(in_features=64, out_features=proj_dim, bias=False)
        
        self.temperature = nn.Parameter(torch.tensor([0.07]), requires_grad=True)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=n_cls)
        )


    def forward(self, x_accel, x_gyro, return_feat=False):
        f_accel = self.enc_A(x_accel)
        f_gyro = self.enc_G(x_gyro)

        # Frequency domain transformation
        accel_freq = torch.fft.fft(x_accel, dim=2)
        gyro_freq = torch.fft.fft(x_gyro, dim=2)
        accel_magnitude = torch.abs(accel_freq)
        gyro_magnitude = torch.abs(gyro_freq)
        # Extract frequency domain features
        f_accel2 = self.enc_A(accel_magnitude)
        f_gyro2 = self.enc_G(gyro_magnitude)

        fused_feat2 = torch.stack((f_accel, f_accel2), dim=0)  # Form Transformer inputs (seq_1en, batch, feature-dim)
        fused_feat2 = self.transformer_encoder(fused_feat2)  # Transformer code
        fused_feat2 = fused_feat2.mean(dim=0)  # Take the average fusion feature

        fused_feat3 = torch.stack((f_gyro, f_gyro2), dim=0)
        fused_feat3 = self.transformer_encoder(fused_feat3)
        fused_feat3 = fused_feat3.mean(dim=0)

        out =  self.classifier(torch.cat((fused_feat2, fused_feat3), dim=-1))
        # out =  self.classifier(torch.cat((f_accel, f_gyro), dim=-1))
        # out = self.classifier(fused_feat)

        # Two domains of self supervised accelerometers
        if self.proj_dim > 0:
            e_accel2 = self.proj_A(f_accel)
            e_accel2_2 = self.proj_G(f_accel2)
        else:
            e_accel2 = f_accel
            e_accel2_2 = f_accel2
        logits2 = torch.mm(F.normalize(e_accel2), F.normalize(e_accel2_2).T) * torch.exp(self.temperature)

        # Two domains of self supervised 3 pairs of gyroscopes
        if self.proj_dim > 0:
            e_gyro2 = self.proj_A(f_gyro)
            e_gyro2_2 = self.proj_G(f_gyro2)
        else:
            e_gyro2 = f_accel
            e_gyro2_2 = f_accel2
        logits3 = torch.mm(F.normalize(e_gyro2), F.normalize(e_gyro2_2).T) * torch.exp(self.temperature)

        # Self supervision 1 used TF to collect information from accelerometers and gyroscopes
        if self.proj_dim > 0:
            e_accel = self.proj_A(fused_feat2)
            e_gyro = self.proj_G(fused_feat3)
        else:
            e_accel = fused_feat2
            e_gyro = fused_feat3
        logits = torch.mm(F.normalize(e_accel), F.normalize(e_gyro).T) * torch.exp(self.temperature)

        if return_feat:
            return logits, logits2, logits3, out, (F.normalize(e_accel), F.normalize(e_gyro)), (F.normalize(e_accel2), F.normalize(e_accel2_2)), (F.normalize(e_gyro2), F.normalize(e_gyro2_2))
        return logits, logits2, logits3, out
    