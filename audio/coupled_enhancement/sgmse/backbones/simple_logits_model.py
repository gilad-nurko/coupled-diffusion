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
    
class LogitsDenoiser(nn.Module):
    """
    Key insight: noisy_logits should be treated as an OBSERVATION of the target,
    not as a parallel stream to fuse with x_t.
    
    Think of it like: "I'm denoising x_t, and I have a noisy measurement
    (noisy_logits) that tells me approximately where I should end up."
    
    This is more like a Kalman filter / posterior estimation problem:
    p(clean | x_t, noisy_logits, audio, history)
    """
    def __init__(
        self,
        logits_size: int = 51865,
        audio_emb_dim: int = 512,
        cond_dim: int = 512,
        width: int = 768,
        depth: int = 10, # was 6
        nhead_hist: int = 8, # was 4
        nhead_audio: int = 4,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.C = logits_size
        self.cond_dim = cond_dim
        self.use_cls_token = use_cls_token
        self.max_history_steps = 32 # max history length in logits steps
        
        # ===== TIME EMBEDDING =====
        self.t_fourier = GaussianFourierProjection(128, 16.0)
        self.t_proj = nn.Sequential(
            nn.Linear(256, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ===== ENCODE NOISY OBSERVATION =====
        # This is your "measurement" of what the clean logits might be
        self.noisy_obs_encoder = nn.Sequential(
            nn.LayerNorm(logits_size),
            nn.Linear(logits_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # # ===== ENCODE AUDIO CONTEXT =====
        # self.audio_proj = nn.Sequential(
        #     nn.Linear(audio_emb_dim, cond_dim),
        #     nn.SiLU(),
        #     nn.Linear(cond_dim, cond_dim)
        # )
        # # Simple temporal pooling with attention
        # self.audio_attn_pool = nn.MultiheadAttention(
        #     embed_dim=cond_dim, num_heads=8, batch_first=True
        # )
        # self.audio_query = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)

        # =========================
        # Audio sequence encoder: [B, T, D_a] -> [B, cond_dim]
        # =========================
        # Per-frame projection to cond space
        self.a_step_proj = nn.Sequential(
            nn.Linear(audio_emb_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        # Temporal encoder over frames
        self.audio_enc = TinyTransformerEnc(
            d_model=cond_dim,
            nhead=nhead_audio,
            num_layers=1,
            dim_feedforward=2 * cond_dim,
            dropout=0.1,
        )
        # Attention pooling: query built from (t_vec + h_vec)
        self.audio_pool_q = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        self.audio_mha = nn.MultiheadAttention(
            embed_dim=cond_dim, num_heads=nhead_audio, batch_first=True
        )
        
        # # ===== ENCODE HISTORY =====
        # self.history_proj = nn.Sequential(
        #     nn.LayerNorm(logits_size),
        #     nn.Linear(logits_size, cond_dim),
        #     nn.SiLU(),
        #     nn.Linear(cond_dim, cond_dim)
        # )

        # =========================
        # History encoder (FULL LOGITS): [B, S, C] -> [B, cond_dim]
        # =========================
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
            # Learnable CLS vector in cond space for robust pooling
            self.hist_cls = nn.Parameter(torch.zeros(cond_dim))
        
        # ===== CONDITIONING FUSION =====
        # Combine [time, noisy_obs, audio, history]
        self.cond_fusion = nn.Sequential(
            nn.Linear(4 * cond_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.LayerNorm(width)
        )
        # self.cond_fusion = nn.Sequential(
        #     nn.Linear(2 * cond_dim, width),
        #     nn.SiLU(),
        #     nn.Linear(width, width),
        #     nn.LayerNorm(width)
        # )
        # Create separate projections for each block
        # self.cond_projections = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(4 * cond_dim, width),
        #         nn.SiLU(),
        #         nn.Linear(width, width)
        #     ) for _ in range(depth)
        # ])

        # # Keep initial fusion too
        # self.initial_cond_fusion = nn.Sequential(
        #     nn.Linear(4 * cond_dim, width),
        #     nn.SiLU(),
        #     nn.Linear(width, width),
        #     nn.LayerNorm(width)
        # )
        
        # ===== INPUT PROJECTION =====
        # Project x_t (the diffusion-noised state)
        self.x_proj = nn.Sequential(
            nn.Linear(logits_size, width),
            nn.LayerNorm(width)
        )
        
        # ===== RESIDUAL BLOCKS WITH ADAPTIVE LAYER NORM =====
        self.blocks = nn.ModuleList([
            AdaLNResBlock(width, width, dropout) for _ in range(depth)
        ])
        
        # ===== OUTPUT =====
        self.out_norm = nn.LayerNorm(width)
        self.out_head = nn.Linear(width, logits_size)
        
        # Initialize output layer to near-zero for stability
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)
    
    def _encode_history(self, history, device):
        """
        history:
        - FloatTensor logits history [B, S_hist, C]  (preferred)
        - (Backward-compat) LongTensor ids [B, S_hist] or ragged List[List[int]]
        Returns:
        - hist_vec: [B, cond_dim]
        """
        # ----- branch A: full logits history [B, S, C] -----
        if isinstance(history, torch.Tensor) and history.dtype.is_floating_point:
            hist_logits = history.to(device)  # [B, S, C]
            B, S, C = hist_logits.shape
            # project each step's logits to cond space
            h = self.logits_hist_proj(hist_logits)   # [B, S, cond_dim]

            key_pad = None  # if you have a padding mask, pass it via a wrapper or store on self

            # prepend CLS in cond space if requested
            if self.use_cls_token:
                cls = self.hist_cls[None, None, :].expand(B, 1, -1)  # [B,1,cond_dim]
                h = torch.cat([cls, h], dim=1)                       # [B, S+1, cond_dim]
                if key_pad is not None:
                    pad_cls = torch.zeros((B, 1), dtype=torch.bool, device=device)
                    key_pad = torch.cat([pad_cls, key_pad], dim=1)

            # transformer encode
            h = self.hist_enc(h, key_padding_mask=key_pad)           # [B, S', cond_dim]

            # pool
            if self.use_cls_token:
                hist_vec = h[:, 0, :]                                # [B, cond_dim]
            else:
                if key_pad is None:
                    hist_vec = h.mean(dim=1)
                else:
                    mask = ~key_pad                                   # True = keep
                    hist_vec = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return hist_vec

        # ----- branch B: ids path (kept for compatibility) -----
        if isinstance(history, list):
            B = len(history)
            lengths = [len(x) for x in history]
            S = max(1, max(lengths)) if len(lengths) > 0 else 1
            pad_id = 0
            ids = torch.full((B, S), fill_value=pad_id, dtype=torch.long, device=device)
            for b, seq in enumerate(history):
                if len(seq) > 0:
                    tlen = min(len(seq), S)
                    ids[b, :tlen] = torch.tensor(seq[:tlen], dtype=torch.long, device=device)
            key_pad = (torch.arange(S, device=device)[None, :] >= torch.tensor(lengths, device=device)[:, None])
        else:
            ids = history.to(device)          # [B, S]
            key_pad = None

        # token-id route (old behavior)
        if self.use_cls_token:
            B, S = ids.shape
            cls_id = int(self.cls_id.item())
            cls = torch.full((B, 1), cls_id, dtype=torch.long, device=device)
            ids = torch.cat([cls, ids], dim=1)
            if key_pad is not None:
                pad_cls = torch.zeros((B, 1), dtype=torch.bool, device=device)
                key_pad = torch.cat([pad_cls, key_pad], dim=1)

        h = self.tok_emb(ids)                 # [B, S', token_emb_dim]
        h = self.hist_proj(h)                 # [B, S', cond_dim]
        h = self.hist_enc(h, key_padding_mask=key_pad)  # [B, S', cond_dim]

        if self.use_cls_token:
            hist_vec = h[:, 0, :]
        else:
            if key_pad is None:
                hist_vec = h.mean(dim=1)
            else:
                mask = ~key_pad
                hist_vec = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return hist_vec
        
    def forward(
        self,
        x_t: torch.Tensor,              # [B, C] - diffusion noised
        noisy_logits: torch.Tensor,     # [B, C] - whisper output on noisy audio
        t: torch.Tensor,                # [B] or scalar
        audio_emb: torch.Tensor,        # [B, T, D_a]
        history_logits: torch.Tensor    # [B, S, C]
    ):
        """
        Returns: predicted noise ε (not score)
        Train with: loss = MSE(predicted_noise, true_noise)
        """
        B = x_t.shape[0]
        device = x_t.device
        
        # ===== TIME EMBEDDING =====
        if t.dim() == 0:
            t = t.expand(B)
        t_emb = self.t_fourier(torch.log(t.float() + 1e-8))
        t_vec = self.t_proj(t_emb)  # [B, cond_dim]
        
        # ===== ENCODE NOISY OBSERVATION =====
        # This is crucial: treat it as valuable information about the target
        obs_vec = self.noisy_obs_encoder(noisy_logits)  # [B, cond_dim]
        
        # # ===== ENCODE AUDIO =====
        # # Project each frame
        # a_proj = self.audio_proj(audio_emb)  # [B, T, cond_dim]
        # # Attention pooling
        # query = self.audio_query.expand(B, -1, -1)  # [B, 1, cond_dim]
        # a_pooled, _ = self.audio_attn_pool(query, a_proj, a_proj)  # [B, 1, cond_dim]
        # a_vec = a_pooled.squeeze(1)  # [B, cond_dim]

        # ===== 1) ENCODE HISTORY =====
        # history will be just the last self.max_history_steps steps of logits
        if history_logits.size(1) > self.max_history_steps:
            history_logits = history_logits[:, -self.max_history_steps:, :]
        h_vec = self._encode_history(history_logits, device)               # [B, cond_dim]

        # =========================
        # 2) Audio sequence encoding & attention pooling
        # =========================
        audio_kpm = None  # key padding mask (True = pad)
        if isinstance(audio_emb, tuple):
            audio_seq, audio_lens = audio_emb
            audio_seq = audio_seq.to(device)                               # [B, T, D_a]
            if audio_lens is not None:
                B_, T_ = audio_seq.shape[:2]
                rng = torch.arange(T_, device=device)[None, :]
                audio_kpm = rng >= audio_lens.to(device)[:, None]          # [B, T] True=pad
        else:
            audio_seq = audio_emb.to(device)                    # [B, T, D_a]

        # per-frame projection then temporal encoder
        a_steps = self.a_step_proj(audio_seq)                               # [B, T, cond_dim]
        a_enc   = self.audio_enc(a_steps, key_padding_mask=audio_kpm)       # [B, T, cond_dim]

        # attention pool with a single query conditioned on h_vec
        q = self.audio_pool_q(h_vec).unsqueeze(1)                   # [B, 1, cond_dim]
        # MultiheadAttention with batch_first=True expects [B, S, E]
        a_pooled, _ = self.audio_mha(q, a_enc, a_enc, key_padding_mask=audio_kpm)  # [B,1,cond_dim]
        a_vec = a_pooled.squeeze(1)                                         # [B, cond_dim]
        
        # # ===== ENCODE HISTORY =====
        # if history_logits.size(1) > 0:
        #     h_proj = self.history_proj(history_logits)  # [B, S, cond_dim]
        #     h_vec = h_proj.mean(dim=1)  # [B, cond_dim]
        # else:
        #     h_vec = torch.zeros(B, self.cond_dim, device=device)
        
        # ===== FUSE ALL CONDITIONING =====
        # Order: [time, observation, audio, history]
        # Time is first because it controls the denoising strength
        # Observation is second because it's our measurement
        # cond_cat = torch.cat([t_vec, obs_vec], dim=-1)  # [B, 2*cond_dim]
        cond_cat = torch.cat([t_vec, obs_vec, a_vec, h_vec], dim=-1)  # [B, 4*cond_dim]
        cond_vec = self.cond_fusion(cond_cat)  # [B, width]
        
        # ===== PROCESS X_T =====
        h = self.x_proj(x_t)  # [B, width]
        # cond_vec = self.initial_cond_fusion(cond_cat)  # [B, width]
        # h = h + cond_vec  # Additive injection
        
        # ===== RESIDUAL BLOCKS =====
        for i, block in enumerate(self.blocks):
            # Re-project conditioning for this depth
            # block_cond = self.cond_projections[i](cond_cat)
            # h = block(h, block_cond)
            h = block(h, cond_vec)
        
        # ===== OUTPUT: PREDICT NOISE =====
        h = self.out_norm(h)
        predicted_noise = self.out_head(h)  # [B, C]
        
        return predicted_noise


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


class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size: int = 128, scale: float = 16.0):
        super().__init__()
        W = torch.randn(embedding_size) * scale
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x_proj = t[:, None] * self.W[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)