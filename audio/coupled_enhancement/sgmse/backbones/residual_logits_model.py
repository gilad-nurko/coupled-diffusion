import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


# -------- tiny Transformer encoder for history --------
class TinyTransformerEnc(nn.Module):
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 4*256, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=dim_feedforward, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, S_hist, d_model], key_padding_mask: [B, S_hist] (True = pad)
        return self.enc(x, src_key_padding_mask=key_padding_mask)  # [B, S_hist, d_model]
    
class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 128, scale: float = 16.0):
        super().__init__()
        W = torch.randn(embedding_size) * scale
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x_proj = t[:, None] * self.W[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class AdaLNResBlock(nn.Module):
    """
    Residual block with Adaptive Layer Norm (like DiT).
    The conditioning modulates the layer norm parameters.
    """
    def __init__(self, width: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # Main path
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)
        
        # Conditioning -> scale and shift for adaptive norm
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * width)
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x: [B, width]
        cond: [B, cond_dim]
        """
        # Get adaptive parameters
        ada_params = self.ada_proj(cond)  # [B, 2*width]
        scale, shift = ada_params.chunk(2, dim=-1)  # [B, width], [B, width]
        
        # Residual path with adaptive norm
        residual = x
        x = self.norm1(x)
        x = x * (1 + scale) + shift  # Adaptive modulation
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return residual + x

class LogitsDenoiser(nn.Module):
    def __init__(
        self,
        logits_size: int = 51865,
        audio_emb_dim: int = 512,
        cond_dim: int = 512,
        width: int = 768,
        depth: int = 6, # was 3
        nhead_hist: int = 8, # was 4
        nhead_audio: int = 4, # was 2
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.C = logits_size
        self.cond_dim = cond_dim
        self.use_cls_token = use_cls_token
        self.max_history_steps = 32
        
        # ===== TIME EMBEDDING =====
        self.t_fourier = GaussianFourierProjection(128, 16.0)
        self.t_proj = nn.Sequential(
            nn.Linear(256, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ===== ENCODE THE RESIDUAL (x_t - noisy_logits) =====
        # This is what goes into the baseline
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(logits_size),
            nn.Linear(logits_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ===== ENCODE NOISY OBSERVATION =====
        self.noisy_obs_encoder = nn.Sequential(
            nn.LayerNorm(logits_size),
            nn.Linear(logits_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ===== AUDIO ENCODING =====
        self.a_step_proj = nn.Sequential(
            nn.Linear(audio_emb_dim, cond_dim), 
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.audio_enc = TinyTransformerEnc(
            d_model=cond_dim,
            nhead=nhead_audio,
            num_layers=1,
            dim_feedforward=2 * cond_dim,
            dropout=0.1,
        )
        self.audio_pool_q = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), 
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.audio_mha = nn.MultiheadAttention(
            embed_dim=cond_dim, num_heads=nhead_audio, batch_first=True
        )
        
        # ===== HISTORY ENCODING =====
        self.logits_hist_proj = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.Linear(self.C, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.hist_enc = TinyTransformerEnc(
            d_model=cond_dim,
            nhead=nhead_hist,
            num_layers=1,
            dim_feedforward=2 * cond_dim,
            dropout=0.1,
        )
        if self.use_cls_token:
            self.hist_cls = nn.Parameter(torch.zeros(cond_dim))
        
        # ===== CONDITIONING FUSION =====
        # [time, residual_info, noisy_obs, audio, history]
        self.cond_fusion = nn.Sequential(
            nn.Linear(5 * cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width)
        )
        
        # ===== INPUT PROJECTION =====
        self.x_proj = nn.Sequential(
            nn.Linear(logits_size, width),
            nn.LayerNorm(width)
        )
        
        # ===== RESIDUAL BLOCKS =====
        self.blocks = nn.ModuleList([
            AdaLNResBlock(width, width, dropout) for _ in range(depth)
        ])
        
        # ===== OUTPUT HEAD FOR CORRECTION =====
        self.out_norm = nn.LayerNorm(width)
        self.correction_head = nn.Linear(width, logits_size)
        
        # Initialize to output near-zero (trust baseline initially)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        
        # # Optional: learnable scale for corrections
        self.baseline_weight_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim // 2),
            nn.SiLU(),
            nn.Linear(cond_dim // 2, 1),
            nn.Sigmoid()  # Weight in [0,1]
        )
    
    def _encode_history(self, history, device):
        if isinstance(history, torch.Tensor) and history.dtype.is_floating_point:
            hist_logits = history.to(device)
            B, S, C = hist_logits.shape
            h = self.logits_hist_proj(hist_logits)
            
            if self.use_cls_token:
                cls = self.hist_cls[None, None, :].expand(B, 1, -1)
                h = torch.cat([cls, h], dim=1)
            
            h = self.hist_enc(h, key_padding_mask=None)
            
            if self.use_cls_token:
                hist_vec = h[:, 0, :]
            else:
                hist_vec = h.mean(dim=1)
            return hist_vec
        
        return torch.zeros(history.shape[0], self.cond_dim, device=device)
    
    def forward(
        self,
        x_t: torch.Tensor,              # [B, C]
        noisy_logits: torch.Tensor,     # [B, C]
        std_l: torch.Tensor,            # [B] or [B, 1]
        audio_emb: torch.Tensor,        # [B, T, D_a]
        history_logits: torch.Tensor    # [B, S, C]
    ):
        """
        Returns: predicted noise (baseline + learned correction)
        
        The model:
        1. Computes baseline: (x_t - noisy_logits) / std_l
        2. Predicts a correction based on audio/history
        3. Returns: baseline + correction
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # ===== COMPUTE BASELINE NOISE PREDICTION =====
        residual = x_t - noisy_logits  # [B, C]
        if std_l.dim() == 1:
            std_l = std_l.unsqueeze(-1)  # [B, 1]
        baseline_score = - residual / (std_l ** 2)  # [B, C]
        
        # ===== TIME EMBEDDING =====
        # Extract t from std_l if needed
        t = std_l.squeeze()
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t_emb = self.t_fourier(torch.log(t.float() + 1e-8))
        t_vec = self.t_proj(t_emb)  # [B, cond_dim]
        
        # ===== ENCODE THE RESIDUAL =====
        # Give the network info about what went into the baseline
        residual_vec = self.residual_encoder(residual)  # [B, cond_dim]
        
        # ===== ENCODE NOISY OBSERVATION =====
        obs_vec = self.noisy_obs_encoder(noisy_logits)  # [B, cond_dim]
        
        # ===== ENCODE HISTORY =====
        if history_logits.size(1) > self.max_history_steps:
            history_logits = history_logits[:, -self.max_history_steps:, :]
        h_vec = self._encode_history(history_logits, device)  # [B, cond_dim]
        
        # ===== ENCODE AUDIO =====
        audio_kpm = None
        if isinstance(audio_emb, tuple):
            audio_seq, audio_lens = audio_emb
            audio_seq = audio_seq.to(device)
            if audio_lens is not None:
                B_, T_ = audio_seq.shape[:2]
                rng = torch.arange(T_, device=device)[None, :]
                audio_kpm = rng >= audio_lens.to(device)[:, None]
        else:
            audio_seq = audio_emb.to(device)
        
        a_steps = self.a_step_proj(audio_seq)
        a_enc = self.audio_enc(a_steps, key_padding_mask=audio_kpm)
        q = self.audio_pool_q(h_vec).unsqueeze(1)
        a_pooled, _ = self.audio_mha(q, a_enc, a_enc, key_padding_mask=audio_kpm)
        a_vec = a_pooled.squeeze(1)
        
        # ===== FUSE CONDITIONING =====
        cond_cat = torch.cat([t_vec, residual_vec, obs_vec, a_vec, h_vec], dim=-1)
        cond_vec = self.cond_fusion(cond_cat)  # [B, width]
        
        # ===== PROCESS RESIDUAL THROUGH NETWORK =====
        h = self.x_proj(residual)  # [B, width]
        
        for block in self.blocks:
            h = block(h, cond_vec)
        
        # ===== PREDICT CORRECTION =====
        h = self.out_norm(h)
        correction = self.correction_head(h)  # [B, C]
        
        # ===== FINAL PREDICTION = BASELINE + CORRECTION =====
        baseline_weight = self.baseline_weight_mlp(t_vec)  # [B, 1]
        predicted_score = baseline_weight * baseline_score + correction
        
        return predicted_score