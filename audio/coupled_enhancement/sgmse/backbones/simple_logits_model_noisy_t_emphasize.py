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
        return self.enc(x, src_key_padding_mask=key_padding_mask)


class LogitsDenoiser(nn.Module):
    """
    Hierarchical design:
    PRIMARY: x_t + noisy_logits (observation) + t → denoising
    AUXILIARY: audio + history → contextual modulation
    
    The noisy_logits act as a "measurement" that's fused early with x_t,
    while audio/history provide lighter conditioning.
    """
    def __init__(
        self,
        logits_size: int = 51865,
        audio_emb_dim: int = 512,
        cond_dim: int = 512,
        width: int = 768,
        depth: int = 6,
        nhead_hist: int = 4,
        nhead_audio: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.C = logits_size
        self.cond_dim = cond_dim
        self.use_cls_token = use_cls_token
        self.max_history_steps = 32
        
        # ===== PRIMARY PATH: TIME EMBEDDING =====
        self.t_fourier = GaussianFourierProjection(128, 16.0)
        self.t_proj = nn.Sequential(
            nn.Linear(256, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ===== PRIMARY PATH: X_T + NOISY_LOGITS FUSION =====
        # Key insight: fuse the observation directly with x_t EARLY
        # This creates a "measurement-aware" representation
        self.x_proj = nn.Linear(logits_size, width // 2)
        self.obs_proj = nn.Linear(logits_size, width // 2)
        
        # Combined input normalization
        self.input_norm = nn.LayerNorm(width)
        
        # ===== AUXILIARY: AUDIO CONTEXT =====
        self.a_step_proj = nn.Sequential(
            nn.Linear(audio_emb_dim, cond_dim // 2),
            nn.SiLU(),
        )
        self.audio_enc = TinyTransformerEnc(
            d_model=cond_dim // 2,
            nhead=max(1, nhead_audio // 2),
            num_layers=1,
            dim_feedforward=cond_dim,
            dropout=0.1,
        )
        self.audio_pool_q = nn.Linear(cond_dim, cond_dim // 2)
        self.audio_mha = nn.MultiheadAttention(
            embed_dim=cond_dim // 2, num_heads=max(1, nhead_audio // 2), batch_first=True
        )
        
        # ===== AUXILIARY: HISTORY CONTEXT =====
        self.logits_hist_proj = nn.Sequential(
            nn.LayerNorm(self.C),
            nn.Linear(self.C, cond_dim // 2),
            nn.SiLU(),
        )
        self.hist_enc = TinyTransformerEnc(
            d_model=cond_dim // 2,
            nhead=max(1, nhead_hist // 2),
            num_layers=1,
            dim_feedforward=cond_dim,
            dropout=0.1,
        )
        if self.use_cls_token:
            self.hist_cls = nn.Parameter(torch.zeros(cond_dim // 2))
        
        # ===== AUXILIARY FUSION (simplified) =====
        # Just combine audio + history as light context
        self.aux_fusion = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
        )
        
        # ===== PRIMARY CONDITIONING =====
        # Combine time + auxiliary context
        self.primary_cond = nn.Sequential(
            nn.Linear(cond_dim + cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        
        # ===== DENOISING BACKBONE =====
        self.blocks = nn.ModuleList([
            AdaLNResBlock(width, width, dropout) for _ in range(depth)
        ])
        
        # ===== OUTPUT =====
        self.out_norm = nn.LayerNorm(width)
        self.out_head = nn.Linear(width, logits_size)
        
        # Initialize output to near-zero
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
    
    def _encode_history(self, history, device):
        """Encode history logits to context vector."""
        if isinstance(history, torch.Tensor) and history.dtype.is_floating_point:
            hist_logits = history.to(device)
            B, S, C = hist_logits.shape
            
            h = self.logits_hist_proj(hist_logits)
            
            if self.use_cls_token:
                cls = self.hist_cls[None, None, :].expand(B, 1, -1)
                h = torch.cat([cls, h], dim=1)
            
            h = self.hist_enc(h)
            
            if self.use_cls_token:
                hist_vec = h[:, 0, :]
            else:
                hist_vec = h.mean(dim=1)
            
            return hist_vec
        
        # Fallback for empty history
        return torch.zeros(history.shape[0] if isinstance(history, torch.Tensor) else len(history), 
                          self.cond_dim // 2, device=device)
        
    def forward(
        self,
        x_t: torch.Tensor,              # [B, C] - diffusion noised
        noisy_logits: torch.Tensor,     # [B, C] - observation
        t: torch.Tensor,                # [B] or scalar
        audio_emb: torch.Tensor,        # [B, T, D_a]
        history_logits: torch.Tensor    # [B, S, C]
    ):
        """
        Hierarchical processing:
        1. PRIMARY: Fuse x_t + noisy_logits early (they're both in logit space)
        2. AUXILIARY: Encode audio + history separately (lighter weight)
        3. COMBINE: Use time + aux to condition the denoising of primary
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # ===== TIME ENCODING =====
        if t.dim() == 0:
            t = t.expand(B)
        t_emb = self.t_fourier(torch.log(t.float() + 1e-8))
        t_vec = self.t_proj(t_emb)  # [B, cond_dim]
        
        # ===== PRIMARY PATH: FUSE X_T + OBSERVATION EARLY =====
        # This is the key change: treat noisy_logits as integral to x_t
        x_features = self.x_proj(x_t)              # [B, width//2]
        obs_features = self.obs_proj(noisy_logits) # [B, width//2]
        
        # Concatenate and normalize together
        h = torch.cat([x_features, obs_features], dim=-1)  # [B, width]
        h = self.input_norm(h)
        
        # ===== AUXILIARY: AUDIO CONTEXT =====
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
        
        q = self.audio_pool_q(t_vec).unsqueeze(1)  # Use time to build query
        a_pooled, _ = self.audio_mha(q, a_enc, a_enc, key_padding_mask=audio_kpm)
        a_vec = a_pooled.squeeze(1)  # [B, cond_dim//2]
        
        # ===== AUXILIARY: HISTORY CONTEXT =====
        if history_logits.size(1) > self.max_history_steps:
            history_logits = history_logits[:, -self.max_history_steps:, :]
        h_vec = self._encode_history(history_logits, device)  # [B, cond_dim//2]
        
        # ===== COMBINE AUXILIARY CONTEXTS =====
        aux_vec = self.aux_fusion(torch.cat([a_vec, h_vec], dim=-1))  # [B, cond_dim]
        
        # ===== PRIMARY CONDITIONING: TIME + AUXILIARY =====
        cond_vec = self.primary_cond(torch.cat([t_vec, aux_vec], dim=-1))  # [B, width]
        
        # ===== DENOISING BACKBONE =====
        for block in self.blocks:
            h = block(h, cond_vec)
        
        # ===== OUTPUT: PREDICT NOISE =====
        h = self.out_norm(h)
        predicted_noise = self.out_head(h)
        
        return predicted_noise


class AdaLNResBlock(nn.Module):
    """Residual block with Adaptive Layer Norm."""
    def __init__(self, width: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)
        self.dropout = nn.Dropout(dropout)
        
        self.ada_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * width)
        )
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        ada_params = self.ada_proj(cond)
        scale, shift = ada_params.chunk(2, dim=-1)
        
        residual = x
        x = self.norm1(x)
        x = x * (1 + scale) + shift
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return residual + x


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 128, scale: float = 16.0):
        super().__init__()
        W = torch.randn(embedding_size) * scale
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x_proj = t[:, None] * self.W[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)