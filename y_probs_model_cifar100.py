import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


class ResNetEncoder(nn.Module):
    """Enhanced ResNet18 with spatial feature preservation"""
    def __init__(self, feature_dim: int = 512, in_channels: int = 3):
        super().__init__()
        backbone = resnet18(pretrained=False)
        backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        
        # Extract layers up to avgpool
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        # 512 is the output of ResNet18's final layer
        self.proj = nn.Linear(512, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 512, 4, 4]
        
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.proj(x)  # [B, feature_dim]
        return x


class SinusoidalPosEmb(nn.Module):
    """Better time embedding than learned embeddings"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ConditionalLinear(nn.Module):
    """Enhanced FiLM conditioning"""
    def __init__(self, num_in: int, num_out: int, time_dim: int):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        
        # Time MLP for FiLM parameters
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, num_out * 4),
            nn.SiLU(),
            nn.Linear(num_out * 4, num_out * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        time_out = self.time_mlp(t_emb)
        gamma, beta = time_out.chunk(2, dim=-1)
        return h * (1.0 + gamma) + beta


class CrossAttention(nn.Module):
    """Attention between logit features and image features"""
    def __init__(self, dim: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(context_dim, dim)
        self.to_v = nn.Linear(context_dim, dim)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.num_heads
        
        q = self.to_q(x).reshape(B, -1, h, x.shape[-1] // h).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, h, x.shape[-1] // h).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, h, x.shape[-1] // h).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, -1, x.shape[-1])
        return self.to_out(out)


class ConditionalModel(nn.Module):
    """
    Enhanced denoiser with:
    - Better time conditioning (sinusoidal)
    - Skip connections (U-Net style)
    - Cross-attention between image and logit features
    - Larger capacity
    """
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 1024, 
        n_input_channels: int = 3,
        num_classes: int = 100,
        timesteps: int = 1000,
        num_heads: int = 8,
    ):
        super().__init__()
        self.y_dim = num_classes
        self.hidden_dim = hidden_dim
        
        # Time embedding
        time_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Image encoder
        self.encoder_x = ResNetEncoder(
            feature_dim=feature_dim,
            in_channels=n_input_channels,
        )
        self.x_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Input projection for concatenated logits
        self.y_input = nn.Sequential(
            nn.Linear(2 * self.y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        
        # Downsampling blocks with skip connections
        self.down1 = ConditionalLinear(hidden_dim, hidden_dim, time_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn1 = CrossAttention(hidden_dim, hidden_dim, num_heads)
        
        self.down2 = ConditionalLinear(hidden_dim, hidden_dim, time_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn2 = CrossAttention(hidden_dim, hidden_dim, num_heads)
        
        # Bottleneck
        self.mid1 = ConditionalLinear(hidden_dim, hidden_dim, time_dim)
        self.mid_norm = nn.LayerNorm(hidden_dim)
        self.mid_attn = CrossAttention(hidden_dim, hidden_dim, num_heads)
        self.mid2 = ConditionalLinear(hidden_dim, hidden_dim, time_dim)
        
        # Upsampling blocks with skip connections
        self.up1 = ConditionalLinear(hidden_dim * 2, hidden_dim, time_dim)
        self.up_norm1 = nn.LayerNorm(hidden_dim)
        
        self.up2 = ConditionalLinear(hidden_dim * 2, hidden_dim, time_dim)
        self.up_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.y_dim),
        )

    def forward(
        self,
        y: torch.Tensor,
        t: torch.Tensor,
        y_cond: torch.Tensor = None,
        x: torch.Tensor = None,
    ) -> torch.Tensor:
        assert y_cond is not None and x is not None
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_dim]
        
        # Image features
        x_feat = self.encoder_x(x)
        x_feat = self.x_proj(x_feat)  # [B, hidden_dim]
        x_feat = x_feat.unsqueeze(1)  # [B, 1, hidden_dim] for attention
        
        # Input logits
        h = torch.cat([y, y_cond], dim=-1)
        h = self.y_input(h)  # [B, hidden_dim]
        h_input = h.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Downsample with skip connections
        h1 = self.down1(h, t_emb)
        h1 = self.norm1(h1)
        h1 = F.silu(h1)
        h1_attn = self.attn1(h1.unsqueeze(1), x_feat).squeeze(1)
        h1 = h1 + h1_attn  # [B, hidden_dim]
        
        h2 = self.down2(h1, t_emb)
        h2 = self.norm2(h2)
        h2 = F.silu(h2)
        h2_attn = self.attn2(h2.unsqueeze(1), x_feat).squeeze(1)
        h2 = h2 + h2_attn  # [B, hidden_dim]
        
        # Bottleneck
        h = self.mid1(h2, t_emb)
        h = self.mid_norm(h)
        h = F.silu(h)
        h_mid_attn = self.mid_attn(h.unsqueeze(1), x_feat).squeeze(1)
        h = h + h_mid_attn
        h = self.mid2(h, t_emb)
        h = F.silu(h)
        
        # Upsample with skip connections
        h = torch.cat([h, h2], dim=-1)  # Skip from down2
        h = self.up1(h, t_emb)
        h = self.up_norm1(h)
        h = F.silu(h)
        
        h = torch.cat([h, h1], dim=-1)  # Skip from down1
        h = self.up2(h, t_emb)
        h = self.up_norm2(h)
        h = F.silu(h)
        
        # Output
        out = self.out(h)
        return out