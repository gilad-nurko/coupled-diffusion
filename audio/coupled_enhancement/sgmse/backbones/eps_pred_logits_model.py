import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- sinusoidal time embedding --------
class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for continuous time/noise scalars.
    Outputs [B, 2*embedding_size] via [sin, cos] features.
    """
    def __init__(self, embedding_size: int = 128, scale: float = 16.0):
        super().__init__()
        # Non-trainable frequencies; registered as buffer so they move with .to(device)
        W = torch.randn(embedding_size) * scale
        self.register_buffer("W", W)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] -> [B, 2*embedding_size]
        x_proj = t[:, None] * self.W[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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

# # -------- FiLM residual block --------
# class FiLMResBlock(nn.Module):
#     def __init__(self, width: int, cond_dim: int, dropout: float = 0.0, mlp_mult: float = 3.0):
#         super().__init__()
#         hidden = int(width * mlp_mult)
#         self.norm1 = nn.LayerNorm(width)
#         self.fc1   = nn.Linear(width, hidden)
#         self.fc2   = nn.Linear(hidden, width)
#         self.drop  = nn.Dropout(dropout)
#         self.film  = nn.Linear(cond_dim, width * 2)
#         self.guidance_gate = nn.Linear(cond_dim, 1)

#     def forward(self, h: torch.Tensor, cond: torch.Tensor,
#                 guided: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """
#         h: [B, W]
#         cond: [B, cond_dim]
#         guided: optional [B, W] (projected guidance path)
#         """
#         B, W = h.shape
#         g = self.guidance_gate(cond).sigmoid()  # [B,1]
#         if guided is not None:
#             h = h + g * guided

#         gamma_beta = self.film(cond)            # [B, 2W]
#         gamma, beta = gamma_beta[:, :W], gamma_beta[:, W:]  # [B, W], [B, W]

#         x = self.norm1(h)
#         x = x * (1.0 + gamma) + beta
#         x = F.gelu(self.fc1(x))
#         x = self.drop(x)
#         x = self.fc2(x)
#         return h + x
    
# -------- FiLM residual block --------
class FiLMDeepConcatResBlock(nn.Module):
    """
    NCSN++-inspired: time (temb) dominates; aux (a_vec,h_vec) is weak & gated.
    Inputs:
      h_x, h_y : [B, W]
      temb     : [B, D]   (from t only; strong)
      aux      : [B, D]   (from audio/history; weak)
    """
    def __init__(self, width: int, cond_dim: int, dropout: float = 0.0, mlp_mult: float = 3.0):
        super().__init__()
        hidden = int(width * mlp_mult)

        # norms + fuse x/y
        self.nx = nn.LayerNorm(width)
        self.ny = nn.LayerNorm(width)
        self.fuse_xy = nn.Linear(2 * width, width)

        # ----- time-dominant FiLM (like NCSN++ temb path) -----
        self.t_to_affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * width)  # -> gamma_t, beta_t
        )

        # ----- weak auxiliary FiLM (audio/history) with small init -----
        self.aux_to_affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * width, bias=True)
        )
        # small weights & negative bias => starts near zero contribution
        nn.init.zeros_(self.aux_to_affine[1].weight)
        nn.init.constant_(self.aux_to_affine[1].bias, 0.0)

        # gates to mix aux, initialized to favor time
        self.aux_gate = nn.Linear(cond_dim, 1)
        nn.init.constant_(self.aux_gate.bias, -4.0)  # sigmoid(-4) ~ 0.018 => tiny at start
        nn.init.zeros_(self.aux_gate.weight)

        # trunk
        self.trunk_fc1 = nn.Linear(width, hidden)
        self.trunk_fc2 = nn.Linear(hidden, width)
        self.drop = nn.Dropout(dropout)

        # split update to both streams
        self.to_dx = nn.Linear(width, width)
        self.to_dy = nn.Linear(width, width)

        # residual write gates from time (optional, keeps stability)
        self.gate_x = nn.Linear(cond_dim, 1)
        self.gate_y = nn.Linear(cond_dim, 1)
        nn.init.constant_(self.gate_x.bias, 1.5)  # ~0.82
        nn.init.zeros_(self.gate_x.weight)
        nn.init.constant_(self.gate_y.bias, 1.5)
        nn.init.zeros_(self.gate_y.weight)

    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor, temb: torch.Tensor, aux: torch.Tensor):
        # concat normalized streams
        fused_in = torch.cat([self.nx(h_x), self.ny(h_y)], dim=-1)   # [B, 2W]
        z = self.fuse_xy(fused_in)                                   # [B, W]

        # FiLM from time (dominant)
        gamma_t, beta_t = self.t_to_affine(temb).chunk(2, dim=-1)    # [B,W],[B,W]

        # FiLM from aux (weak, gated)
        aux_strength = torch.sigmoid(self.aux_gate(aux))             # [B,1] ~ 0 initially
        gamma_a, beta_a = self.aux_to_affine(aux).chunk(2, dim=-1)   # [B,W],[B,W]
        gamma = gamma_t + aux_strength * gamma_a
        beta  = beta_t  + aux_strength * beta_a

        # apply FiLM
        z = z * (1.0 + gamma) + beta

        # trunk -> updates
        u = F.gelu(self.trunk_fc1(z))
        u = self.drop(u)
        u = self.trunk_fc2(u)

        dx = self.to_dx(u)
        dy = self.to_dy(u)

        gx = torch.sigmoid(self.gate_x(temb))  # time-controlled write
        gy = torch.sigmoid(self.gate_y(temb))

        h_x = h_x + gx * dx
        h_y = h_y + gy * dy
        return h_x, h_y

class CrossStreamAttention(nn.Module):
    """Bidirectional cross-attention between x and y streams"""
    def __init__(self, width: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm_x = nn.LayerNorm(width)
        self.norm_y = nn.LayerNorm(width)
        
        # x attends to y
        self.cross_attn_x = nn.MultiheadAttention(
            width, nhead, dropout=dropout, batch_first=True
        )
        # y attends to x
        self.cross_attn_y = nn.MultiheadAttention(
            width, nhead, dropout=dropout, batch_first=True
        )
        
        # Optional: gates to control cross-stream influence
        self.gate_x = nn.Linear(width, 1)
        self.gate_y = nn.Linear(width, 1)
        nn.init.constant_(self.gate_x.bias, 0.0)  # Start at ~0.5
        nn.init.constant_(self.gate_y.bias, 0.0)
        
    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor):
        # h_x, h_y: [B, W]
        # Add sequence dimension for MHA
        x_norm = self.norm_x(h_x).unsqueeze(1)  # [B, 1, W]
        y_norm = self.norm_y(h_y).unsqueeze(1)  # [B, 1, W]
        
        # Cross-attention: x queries y
        x_cross, _ = self.cross_attn_x(x_norm, y_norm, y_norm)  # [B, 1, W]
        x_cross = x_cross.squeeze(1)  # [B, W]
        
        # Cross-attention: y queries x
        y_cross, _ = self.cross_attn_y(y_norm, x_norm, x_norm)  # [B, 1, W]
        y_cross = y_cross.squeeze(1)  # [B, W]
        
        # Gated residual
        gx = self.gate_x(h_x).sigmoid()
        gy = self.gate_y(h_y).sigmoid()
        
        h_x = h_x + gx * x_cross
        h_y = h_y + gy * y_cross
        
        return h_x, h_y

class FinalFuseAttention(nn.Module):
    """
    Time-dominant fusion of (h_x, h_y) -> h.
    - Query is t-strong: q = q_t + σ(g(t))*q_aux, with g(t) init ~0.
    - No fallback/averaging; output is pure attention over [h_x, h_y].
    """
    def __init__(self, width: int, cond_dim: int, nhead: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_x = nn.LayerNorm(width)
        self.norm_y = nn.LayerNorm(width)

        # Strong path from time
        self.t_to_q = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(),
            nn.Linear(width, width)
        )

        # Weak path from aux (zero init => near-zero influence at start)
        self.aux_to_q = nn.Sequential(
            nn.Linear(cond_dim, width), nn.SiLU(),
            nn.Linear(width, width)
        )
        nn.init.zeros_(self.aux_to_q[2].weight)
        nn.init.zeros_(self.aux_to_q[2].bias)

        # Gate that decides how much aux affects the query; depends on t
        self.aux_q_gate = nn.Linear(cond_dim, 1)
        nn.init.zeros_(self.aux_q_gate.weight)
        nn.init.constant_(self.aux_q_gate.bias, -4.0)  # σ(-4)≈0.018

        self.mha = nn.MultiheadAttention(embed_dim=width, num_heads=nhead,
                                         dropout=dropout, batch_first=True)

    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor,
                t_vec: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        """
        h_x, h_y : [B, W]
        t_vec    : [B, cond_dim]  (dominant)
        aux      : [B, cond_dim]  (weak, gated)
        returns  : [B, W]
        """
        # Keys/Values are the two tokens
        kv = torch.stack([self.norm_x(h_x), self.norm_y(h_y)], dim=1)  # [B, 2, W]

        # Query: strong time + weak gated aux
        q_t   = self.t_to_q(t_vec)                    # [B, W]
        q_aux = self.aux_to_q(aux)                    # [B, W]
        lam_q = torch.sigmoid(self.aux_q_gate(t_vec)) # [B, 1]
        q = (q_t + lam_q * q_aux).unsqueeze(1)        # [B, 1, W]

        attn_out, _ = self.mha(q, kv, kv)             # [B, 1, W]
        return attn_out.squeeze(1)                    # [B, W]


# -------- Main logits denoiser --------
class LogitsDenoiser(nn.Module):
    """
    Score network s_theta(x_t, t, cond) for Whisper logits (per-step).
    Inputs:
      - x_t_step: [B, C]
      - noisy_step: [B, C]
      - t: [B] or scalar
      - noisy_audio_embedding: [B, D_a]
      - out: ragged list of token ids (allowed-vocab indices) per batch OR
             a padded LongTensor [B, S_hist]
    Returns:
      - score: [B, C]
    """
    def __init__(
        self,
        logits_size: int,         # C = |A| for input/output (also == vocab_size)
        audio_emb_dim: int = 512,       # D_a
        token_emb_dim: int = 256, # kept for API stability (unused)
        cond_dim: int = 384,
        width: int = 768,
        depth: int = 4,
        nhead_hist: int = 4,
        nhead_audio: int = 2,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.C = logits_size
        self.use_cls_token = use_cls_token
        self.max_history_steps = 32 # max history length in logits steps
        self.centered = True  # whether input logits are centered in [-1,1] or [0,1]

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

        # =========================
        # Time embedding -> cond vector
        # =========================
        self.t_fourier = GaussianFourierProjection(embedding_size=128, scale=16.0)  
        self.t_proj = nn.Sequential(                                                
            nn.Linear(256, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

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

        # Strong time MLP (NCSN++-style two-layer)
        self.t_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Weak auxiliary MLP for (a_vec, h_vec)
        self.aux_mlp = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )
        # small init so it starts weak
        nn.init.zeros_(self.aux_mlp[2].weight)
        nn.init.zeros_(self.aux_mlp[2].bias)

        # # =========================
        # # Input projections for logits paths
        # # =========================
        # self.x_proj       = nn.Linear(self.C, width)  # for x_t_step
        # self.guided_proj  = nn.Linear(self.C, width)  # for noisy_step guidance

        # =========================
        # Input projection for logits paths
        # =========================
        self.x_proj_x = nn.Linear(self.C, width)  # x_t_step -> W
        self.x_proj_y = nn.Linear(self.C, width)  # noisy_step -> W

        # =========================
        # FiLM-modulated residual trunk
        # =========================
        self.blocks = nn.ModuleList()
        self.cross_attns = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                FiLMDeepConcatResBlock(width, cond_dim, dropout=dropout, mlp_mult=3.0)
            )
            # Add cross-attention after each FiLM block (or every N blocks)
            if i % 1 == 0:  # Every block, or change to i % 2 for every other block
                self.cross_attns.append(
                    CrossStreamAttention(width, nhead=8, dropout=dropout)
                )
        self.final_fuse = FinalFuseAttention(width=width, cond_dim=cond_dim, nhead=4, dropout=dropout)
        self.out_norm = nn.LayerNorm(width)
        self.out_head = nn.Linear(width, self.C)

        # =========================
        # Init
        # =========================
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

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
        x_t_step: torch.Tensor,             # [B, C]
        noisy_step: torch.Tensor,           # [B, C]
        t: torch.Tensor,                    # [B] or scalar
        noisy_audio_embedding,              # [B, T, D_a]  OR  (Tensor [B,T,D_a], lengths [B])
        history_logits: torch.Tensor        # [B, S, C]
    ) -> torch.Tensor:                      # -> score [B, C]
        # ----- shape checks -----
        B, C = x_t_step.shape
        assert C == self.C, f"Expected logits size C={self.C}, got {C}"
        assert noisy_step.shape == (B, C), "noisy_step must be [B, C]"
        assert isinstance(history_logits, torch.Tensor) and history_logits.dim() == 3, \
            "history_logits must be FloatTensor [B, S, C]"

        device = x_t_step.device

        # =========================
        # 1) Time & history encodings
        # =========================
        # time embedding -> cond_dim via self.t_proj
        t = t.to(device)
        if t.dim() == 0:
            t = t[None].expand(B)  # scalar -> [B]
        t_emb = self.t_fourier(torch.log(t.float()))                 # [B, 256] from Gaussian Fourier
        t_vec = self.t_proj(t_emb)                                       # [B, cond_dim]

        # history will be just the last self.max_history_steps steps of logits
        if history_logits.size(1) > self.max_history_steps:
            history_logits = history_logits[:, -self.max_history_steps:, :]
        h_vec = self._encode_history(history_logits, device)               # [B, cond_dim]

        # =========================
        # 2) Audio sequence encoding & attention pooling
        # =========================
        audio_kpm = None  # key padding mask (True = pad)
        if isinstance(noisy_audio_embedding, tuple):
            audio_seq, audio_lens = noisy_audio_embedding
            audio_seq = audio_seq.to(device)                               # [B, T, D_a]
            if audio_lens is not None:
                B_, T_ = audio_seq.shape[:2]
                rng = torch.arange(T_, device=device)[None, :]
                audio_kpm = rng >= audio_lens.to(device)[:, None]          # [B, T] True=pad
        else:
            audio_seq = noisy_audio_embedding.to(device)                    # [B, T, D_a]

        # per-frame projection then temporal encoder
        a_steps = self.a_step_proj(audio_seq)                               # [B, T, cond_dim]
        a_enc   = self.audio_enc(a_steps, key_padding_mask=audio_kpm)       # [B, T, cond_dim]

        # attention pool with a single query conditioned on h_vec
        q = self.audio_pool_q(h_vec).unsqueeze(1)                   # [B, 1, cond_dim]
        # MultiheadAttention with batch_first=True expects [B, S, E]
        a_pooled, _ = self.audio_mha(q, a_enc, a_enc, key_padding_mask=audio_kpm)  # [B,1,cond_dim]
        a_vec = a_pooled.squeeze(1)                                         # [B, cond_dim]

        # =========================
        # 3) build conditioning vectors
        # =========================
        # ----- build dominant time embedding -----
        t_vec = self.t_mlp(t_vec)                     # [B, cond_dim]

        # ----- build weak auxiliary embedding from audio & history -----
        aux_in = torch.cat([a_vec, h_vec], dim=-1)   # [B, 2*cond_dim]
        aux  = self.aux_mlp(aux_in)                  # [B, cond_dim]

        # =========================
        # 4) Denoiser trunk (FiLM blocks with guided residual)
        # =========================
        if not self.centered:
            # If input data is in [0, 1]
            x_t_step = 2 * x_t_step - 1.
            noisy_step = 2 * noisy_step - 1.

        h_x = self.x_proj_x(x_t_step)                                          # [B, W]
        h_y = self.x_proj_y(noisy_step)                                        # [B, W]

        # --- deep fusion at every block (both streams updated) ---
        cross_attn_idx = 0
        for i, blk in enumerate(self.blocks):
            h_x, h_y = blk(h_x, h_y, temb=t_vec, aux=aux)                      # [B, W]
            # cross-attention between streams
            if i % 1 == 0 and cross_attn_idx < len(self.cross_attns):
                h_x, h_y = self.cross_attns[cross_attn_idx](h_x, h_y)
                cross_attn_idx += 1

        h = self.final_fuse(h_x, h_y, t_vec=t_vec, aux=aux)   # [B, W]                                                 
        h = self.out_norm(h)
        score = self.out_head(h)                                            # [B, C]
        # # scale by sigma
        # sigma = t.view(-1, 1)
        # score = score / sigma
        return score
