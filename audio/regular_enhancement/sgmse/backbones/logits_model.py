import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import math
#########################################
# Modified ResNetEncoder for CIFAR-10
#########################################

class ResNetEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(ResNetEncoder, self).__init__()
        # Load a ResNet-18 backbone and modify it for CIFAR-10:
        backbone = resnet18(pretrained=False)
        # Replace the first conv layer: 3-channel input, 3x3 kernel, stride=1, padding=1.
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Remove the initial max pooling to keep the spatial resolution high.
        backbone.maxpool = nn.Identity()
        
        # Remove the final fully connected layer.
        modules = []
        for name, module in backbone.named_children():
            if name != 'fc':
                modules.append(module)
        self.f = nn.Sequential(*modules)
        
        # Get the feature dimension from the backbone before the FC layer.
        self.featdim = backbone.fc.weight.shape[1]
        # Project to the desired feature_dim.
        self.g = nn.Linear(self.featdim, feature_dim)

    def forward_feature(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        feature = self.g(feature)
        return feature

    def forward(self, x):
        return self.forward_feature(x)

#########################################
# Modified LeNet5 for CIFAR-10
#########################################

class LeNet5(nn.Module):
    def __init__(self, feature_dim=32, n_input_channels=3, n_input_padding=0):
        super(LeNet5, self).__init__()
        # For CIFAR-10: images have shape (3, 32, 32); no padding needed.
        self.layer1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 6, kernel_size=5, stride=1, padding=n_input_padding),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        # For a 32x32 input, after two layers the feature map size becomes:
        # 32 -> (32-5+1)=28 then pooled -> 14; 14 -> (14-5+1)=10 then pooled -> 5.
        # For consistency with your original linear layer size, we assume an intermediate flattening of size 16*5*5=400.
        self.fc0 = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, feature_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc0(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

#########################################
# ConditionalModel using the encoder for x
#########################################

def get_timestep_embedding(t, emb_dim):
    """
    t:   (B,) float in (t_eps, T]
    returns: (B, emb_dim)
    """
    half = emb_dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t.device) * -(math.log(10000.0) / (half - 1))
    )                                           # (half,)
    args = t[:, None] * freqs[None, :]           # (B, half)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb_dim % 2 == 1:                        # odd dim?
        emb = F.pad(emb, (0, 1))
    return emb

# class ConditionalLinear(nn.Module):
#     def __init__(self, num_in, num_out, n_steps):
#         super(ConditionalLinear, self).__init__()
#         self.num_out = num_out
#         self.lin = nn.Linear(num_in, num_out)
#         self.embed = nn.Embedding(n_steps, num_out)
#         self.embed.weight.data.uniform_()

#     def forward(self, x, t):
#         out = self.lin(x)
#         gamma = self.embed(t)
#         out = gamma.view(-1, self.num_out) * out
#         return out

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, time_emb_dim=128):
        super().__init__()
        self.lin      = nn.Linear(num_in, num_out)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, num_out),
            nn.SiLU(),
            nn.Linear(num_out, num_out)
        )

    def forward(self, x, t):
        # t is (B,) float – same as in audio branch
        temb  = get_timestep_embedding(t, self.time_mlp[0].in_features)
        gamma = self.time_mlp(temb)                 # (B, num_out)
        out   = self.lin(x)
        out   = gamma * out                         # FiLM-style modulation
        return out


class ConditionalModel(nn.Module):
    def __init__(self, feature_dim=32, n_input_channels=3, n_input_padding=0, num_classes=10, timesteps=1000):
        super(ConditionalModel, self).__init__()
        y_dim = num_classes
        # Encoder for x updated for CIFAR-10:
        self.norm = nn.BatchNorm1d(feature_dim)
        # Unet-style conditional blocks:
        self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, timesteps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, timesteps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, timesteps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, y, y_cond, t):
        # Concatenate the provided labels/conditioning information.
        y = torch.cat([y, y_cond], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        return self.lin4(y)

class LogitScoreNet(nn.Module):
    """
    Predicts ε/σ (not ε) for noisy logits so that score = -output.
    """
    def __init__(self, num_classes=10, hidden=128, timesteps=1000):
        super().__init__()
        in_dim = num_classes * 2                # y_t  +  y_cond
        self.lin1 = ConditionalLinear(in_dim,   hidden, timesteps)
        self.lin2 = ConditionalLinear(hidden,   hidden, timesteps)
        self.lin3 = ConditionalLinear(hidden,   hidden, timesteps)
        self.lin_out = nn.Linear(hidden, num_classes)

        self.norm1 = nn.GroupNorm(num_groups=hidden//4, num_channels=hidden)
        self.norm2 = nn.GroupNorm(num_groups=hidden//4, num_channels=hidden)
        self.norm3 = nn.GroupNorm(num_groups=hidden//4, num_channels=hidden)
        # self.norm1 = nn.BatchNorm1d(hidden)
        # self.norm2 = nn.BatchNorm1d(hidden)
        # self.norm3 = nn.BatchNorm1d(hidden)

    def forward(self, y_t, y_cond, t):
        if y_cond.dim() == 1:
            y_cond = y_cond.unsqueeze(0)      
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(0)       
        h = torch.cat([y_t, y_cond], dim=-1)         # (B, 2·C)
        h = F.silu(self.norm1(self.lin1(h, t)))
        h = F.silu(self.norm2(self.lin2(h, t)))
        h = F.silu(self.norm3(self.lin3(h, t)))
        return self.lin_out(h)                      # (B, C)  =  ε/σ
