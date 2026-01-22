# coding=utf-8
# Enhanced NCSN++ with advanced logits-audio fusion
# BACKWARD COMPATIBLE: All original weights load identically

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
import numpy as np
import math

from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class AdaptiveLogitsEncoder(nn.Module):
    """
    Hierarchical encoding of logits with attention and uncertainty modeling.
    Captures both token-level and sequence-level patterns.
    """
    def __init__(self, logits_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.logits_dim = logits_dim
        self.hidden_dim = hidden_dim
        
        # Compute uncertainty/entropy from logits distribution
        # This helps model focus on high-confidence vs uncertain predictions
        self.uncertainty_proj = nn.Linear(1, hidden_dim // 4)
        
        # Token distribution encoder
        self.token_encoder = nn.Sequential(
            nn.LayerNorm(logits_dim),
            nn.Linear(logits_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Bi-directional context modeling
        self.forward_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.backward_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        
        # Self-attention over sequence
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Combine uncertainty features
        self.combine_uncertainty = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def compute_entropy(self, logits):
        """Compute entropy of logits distribution as uncertainty measure"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # [B, S, 1]
        return entropy
        
    def forward(self, class_logits):
        """
        Args:
            class_logits: [B, S, |A|]
        Returns:
            encoded: [B, S, output_dim]
            uncertainty: [B, S, 1] - entropy measure
        """
        B, S, A = class_logits.shape
        
        # Compute uncertainty/entropy from distribution
        uncertainty = self.compute_entropy(class_logits)  # [B, S, 1]
        uncertainty_feat = self.uncertainty_proj(uncertainty)  # [B, S, hidden_dim//4]
        
        # Encode token distributions
        x = self.token_encoder(class_logits)  # [B, S, hidden_dim]
        
        # Bidirectional LSTM for temporal context
        forward_out, _ = self.forward_lstm(x)  # [B, S, hidden_dim//2]
        backward_out, _ = self.backward_lstm(x.flip(dims=[1]))  # [B, S, hidden_dim//2]
        backward_out = backward_out.flip(dims=[1])
        x_bi = torch.cat([forward_out, backward_out], dim=-1)  # [B, S, hidden_dim]
        
        # Self-attention over sequence
        attn_out, _ = self.self_attn(x_bi, x_bi, x_bi)  # [B, S, hidden_dim]
        x = self.attn_norm(x_bi + attn_out)
        
        # Incorporate uncertainty
        x = torch.cat([x, uncertainty_feat], dim=-1)  # [B, S, hidden_dim + hidden_dim//4]
        x = self.combine_uncertainty(x)  # [B, S, hidden_dim]
        
        # Final projection
        out = self.output_proj(x)  # [B, S, output_dim]
        
        return out, uncertainty


class AlignmentModule(nn.Module):
    """
    Learns monotonic alignment between logits sequence S and audio time W.
    Uses both soft and hard alignment strategies with attention mechanisms.
    """
    def __init__(self, logits_dim, audio_dim, num_heads=4):
        super().__init__()
        self.logits_dim = logits_dim
        self.audio_dim = audio_dim
        
        # Project to common dimension for alignment
        self.logits_to_align = nn.Linear(logits_dim, audio_dim)
        self.audio_to_align = nn.Conv1d(audio_dim, audio_dim, kernel_size=1)
        
        # Alignment attention with positional bias
        self.alignment_attn = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable position biases for monotonic alignment
        self.register_buffer('position_bias_scale', torch.tensor(2.0))
        
        # # Duration predictor (predicts how many audio frames per token)
        # self.duration_predictor = nn.Sequential(
        #     nn.Conv1d(logits_dim, logits_dim, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(logits_dim),
        #     nn.Dropout(0.1),
        #     nn.Conv1d(logits_dim, 1, kernel_size=1),
        #     nn.Softplus()  # Ensure positive durations
        # )
        
        # nn.init.zeros_(self.duration_predictor[-2].weight)
        # nn.init.zeros_(self.duration_predictor[-2].bias)
        
    def compute_alignment_bias(self, S, W):
        """Compute monotonic position bias matrix [S, W]"""
        s_pos = torch.arange(S, device=self.position_bias_scale.device).float()
        w_pos = torch.arange(W, device=self.position_bias_scale.device).float()
        
        # Normalize positions to [0, 1]
        s_pos = s_pos / (S - 1) if S > 1 else s_pos
        w_pos = w_pos / (W - 1) if W > 1 else w_pos
        
        # Compute distance from diagonal (monotonic alignment)
        # [S, 1] - [1, W] -> [S, W]
        distance = torch.abs(s_pos.unsqueeze(1) - w_pos.unsqueeze(0))
        
        # Convert distance to bias (prefer diagonal)
        bias = -distance * self.position_bias_scale
        return bias
        
    def forward(self, logits_seq, audio_seq, uncertainty=None):
        """
        Args:
            logits_seq: [B, S, D_logits]
            audio_seq: [B, W, D_audio]
            uncertainty: [B, S, 1] - optional uncertainty from logits
        Returns:
            aligned_logits: [B, W, D_audio] - logits aligned to audio timeline
            alignment_weights: [B, W, S] - soft alignment weights
        """
        B, S, D_logits = logits_seq.shape
        B, W, D_audio = audio_seq.shape
        
        # # Predict duration for each token (how many frames it should span)
        # durations = self.duration_predictor(logits_seq.permute(0, 2, 1))  # [B, 1, S]
        # durations = durations.squeeze(1)  # [B, S]
        
        # # Normalize durations to sum to W
        # durations = durations / durations.sum(dim=1, keepdim=True) * W
        
        # Project to common space
        logits_aligned = self.logits_to_align(logits_seq)  # [B, S, D_audio]
        audio_for_attn = self.audio_to_align(audio_seq.permute(0, 2, 1)).permute(0, 2, 1)
        
        # # Compute alignment bias
        # position_bias = self.compute_alignment_bias(S, W)  # [S, W]
        # position_bias = position_bias.t().unsqueeze(0).expand(B, -1, -1)  # [B, W, S]
        
        # Interpolate logits to audio length using durations as guidance
        # Simple linear interpolation as baseline
        logits_interp = logits_aligned.permute(0, 2, 1)  # [B, D, S]
        logits_interp = F.interpolate(logits_interp, size=W, mode='linear', align_corners=False)
        logits_interp = logits_interp.permute(0, 2, 1)  # [B, W, D]
        
        # Soft attention-based alignment with position bias
        # Audio queries attend to logits keys/values
        attn_out, alignment_weights = self.alignment_attn(
            audio_for_attn,
            logits_aligned,
            logits_aligned,
            attn_mask=None,
        )
        
        # Combine interpolated and attention-based alignment
        aligned = 0.5 * logits_interp + 0.5 * attn_out
        
        return aligned, alignment_weights#, durations


class AudioLogitsFusionBlock(nn.Module):
    """
    Advanced fusion module that combines audio and logits with multiple mechanisms:
    1. Cross-attention (audio attends to logits)
    2. Gated fusion
    3. Frequency-aware modulation
    """
    def __init__(self, audio_channels, logits_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.audio_channels = audio_channels
        self.logits_dim = logits_dim
        
        # Alignment module
        self.alignment = AlignmentModule(logits_dim, audio_channels, num_heads)
        
        # Audio preprocessing
        self.audio_norm = nn.GroupNorm(32, audio_channels)
        
        # Cross-attention: audio queries attend to aligned logits
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=audio_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(audio_channels)
        
        # Gated fusion mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(audio_channels * 2, audio_channels),
            nn.LayerNorm(audio_channels),
            nn.Sigmoid()
        )
        
        # Frequency-aware modulation
        # Different frequency bands benefit differently from linguistic info
        self.freq_modulation = nn.Sequential(
            nn.Conv2d(audio_channels, audio_channels // 2, kernel_size=1),
            nn.GroupNorm(16, audio_channels // 2),
            nn.SiLU(),
            nn.Conv2d(audio_channels // 2, audio_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(audio_channels, audio_channels, kernel_size=1)
        
        # Zero init
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
    def forward(self, audio_features, logits_encoded, uncertainty=None):
        """
        Args:
            audio_features: [B, C, H, W] - audio spectrogram features
            logits_encoded: [B, S, D] - encoded logits sequence
            uncertainty: [B, S, 1] - optional uncertainty
        Returns:
            fused_features: [B, C, H, W]
            alignment_weights: [B, W, S]
        """
        B, C, H, W = audio_features.shape
        
        # Normalize audio
        audio_norm = self.audio_norm(audio_features)
        
        # Convert audio to sequence (average over frequency for alignment)
        audio_seq = audio_norm.mean(dim=2)  # [B, C, W]
        audio_seq = audio_seq.permute(0, 2, 1)  # [B, W, C]
        
        # Align logits to audio timeline
        aligned_logits, alignment_weights = self.alignment(   # , durations if want to add them
            logits_encoded, audio_seq, uncertainty
        )  # [B, W, C]
        
        # Cross-attention
        attn_out, cross_attn_weights = self.cross_attn(
            audio_seq,
            aligned_logits,
            aligned_logits
        )  # [B, W, C]
        
        audio_seq_attended = self.attn_norm(audio_seq + attn_out)
        
        # Gated fusion: decide how much to use logits info
        gate_input = torch.cat([audio_seq, audio_seq_attended], dim=-1)  # [B, W, 2C]
        gate = self.gate_net(gate_input)  # [B, W, C]
        
        fused_seq = audio_seq * (1 - gate) + audio_seq_attended * gate  # [B, W, C]
        
        # Reshape back to 2D: [B, W, C] -> [B, C, W] -> [B, C, H, W]
        fused_seq = fused_seq.permute(0, 2, 1)  # [B, C, W]
        fused_2d = fused_seq.unsqueeze(2).expand(-1, -1, H, -1)  # [B, C, H, W]
        
        # Frequency-aware modulation
        freq_mod = self.freq_modulation(audio_features)  # [B, C, H, W]
        fused_2d = fused_2d * freq_mod
        
        # Output projection
        output = self.output_proj(fused_2d)  # [B, C, H, W]
        
        return output, alignment_weights


class ContextAwareGate(nn.Module):
    """
    Context-aware gating that considers:
    1. Noise level (temporal embedding)
    2. Local audio features
    3. Logits uncertainty
    4. Global context
    """
    def __init__(self, temb_dim, spatial_ch):
        super().__init__()
        
        # Noise-level dependent gate
        self.noise_gate = nn.Sequential(
            nn.Linear(temb_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        
        # Spatial uncertainty gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(spatial_ch, spatial_ch // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, spatial_ch // 4),
            nn.SiLU(),
            nn.Conv2d(spatial_ch // 4, spatial_ch // 8, kernel_size=3, padding=1),
            nn.GroupNorm(4, spatial_ch // 8),
            nn.SiLU(),
            nn.Conv2d(spatial_ch // 8, 1, kernel_size=1),
        )
        
        # Global context gate (average pool + MLP)
        self.global_gate = nn.Sequential(
            nn.Linear(spatial_ch, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        
        # Combine all gates
        self.combine_gates = nn.Sequential(
            nn.Linear(3, 8),
            nn.SiLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # Initialize to near-zero
        for module in [self.noise_gate, self.spatial_gate, self.global_gate]:
            if hasattr(module[-1], 'weight'):
                nn.init.zeros_(module[-1].weight)
                nn.init.constant_(module[-1].bias, -3.0)
        
    def forward(self, temb, spatial_features, cond_features, uncertainty=None):
        """
        Args:
            temb: [B, temb_dim]
            spatial_features: [B, C, H, W]
            cond_features: [B, C, H, W]
            uncertainty: [B, S, 1] - optional
        Returns:
            gated_cond: [B, C, H, W]
        """
        B, C, H, W = spatial_features.shape
        
        # Noise-level gate (higher noise = more guidance)
        noise_gate = torch.sigmoid(self.noise_gate(temb))  # [B, 1]
        
        # Spatial gate (uncertain regions get more guidance)
        spatial_gate = torch.sigmoid(self.spatial_gate(spatial_features))  # [B, 1, H, W]
        spatial_gate_scalar = spatial_gate.mean(dim=[2, 3])  # [B, 1]
        
        # Global context gate
        global_context = spatial_features.mean(dim=[2, 3])  # [B, C]
        global_gate = torch.sigmoid(self.global_gate(global_context))  # [B, 1]
        
        # Combine gates
        all_gates = torch.cat([noise_gate, spatial_gate_scalar, global_gate], dim=1)  # [B, 3]
        combined_gate = self.combine_gates(all_gates)  # [B, 1]
        
        # Apply to spatial
        combined_gate = combined_gate.view(B, 1, 1, 1) * spatial_gate  # [B, 1, H, W]
        
        return combined_gate * cond_features


@BackboneRegistry.register("ncsnpp_48k_logits_conditioned")
class NCSNpp_48k_Logits(nn.Module):
    """
    Enhanced NCSN++ with advanced audio-logits fusion.
    
    Key improvements:
    1. Uncertainty-aware logits encoding
    2. Learned monotonic alignment between logits and audio
    3. Multi-mechanism fusion (attention + gating + modulation)
    4. Context-aware adaptive gating
    
    BACKWARD COMPATIBLE: All logits modules stored separately.
    """

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--ch_mult", type=int, nargs='+', default=[1,1,2,2,2,2,2])
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--attn_resolutions", type=int, nargs='+', default=[])
        parser.add_argument("--nf", type=int, default=128)
        parser.add_argument("--no-centered", dest="centered", action="store_false")
        parser.add_argument("--centered", dest="centered", action="store_true")
        parser.add_argument("--progressive", type=str, default='none')
        parser.add_argument("--progressive_input", type=str, default='none')
        parser.add_argument("--logits_hidden_dim", type=int, default=256) 
        parser.add_argument("--logits_cond_scales", type=int, nargs='+', default=[2, 4], 
                          help="Indices of resolutions to apply logits conditioning")
        parser.set_defaults(centered=True)
        return parser

    def __init__(self,
        scale_by_sigma=True,
        nonlinearity='swish',
        nf=128,
        ch_mult=(1, 1, 2, 2, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=(),
        resamp_with_conv=True,
        conditional=True,
        fir=True,
        fir_kernel=[1, 3, 3, 1],
        skip_rescale=True,
        resblock_type='biggan',
        progressive='none',
        progressive_input='none',
        progressive_combine='sum',
        init_scale=0.,
        fourier_scale=16,
        image_size=256,
        embedding_type='fourier',
        dropout=.0,
        centered=True,
        logits_dim=838,
        logits_hidden_dim=256, 
        logits_cond_scales=[2, 4],
        **unused_kwargs
    ):
        super().__init__()
        self.act = act = get_act(nonlinearity)

        self.nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma
        self.logits_dim = logits_dim
        self.logits_cond_scales = logits_cond_scales

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        init_scale = init_scale
        
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        num_channels = 4
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        # ============================================================
        # Advanced logits conditioning modules (stored separately)
        # ============================================================
        logits_output_dim = 128 
        
        # Hierarchical logits encoder with uncertainty
        self.logits_encoder = AdaptiveLogitsEncoder(
            logits_dim=logits_dim,
            hidden_dim=logits_hidden_dim,
            output_dim=logits_output_dim,
            num_heads=4 
        )
        
        # Multi-scale fusion blocks
        self.fusion_blocks = nn.ModuleDict()
        self.context_gates = nn.ModuleDict()
        
        for scale_idx in logits_cond_scales:
            if scale_idx < num_resolutions:
                target_ch = nf * ch_mult[scale_idx]
                
                self.fusion_blocks[f'scale_{scale_idx}'] = AudioLogitsFusionBlock(
                    audio_channels=target_ch,
                    logits_dim=logits_output_dim,
                    num_heads=4,
                    dropout=0.1
                )
                
                self.context_gates[f'scale_{scale_idx}'] = ContextAwareGate(
                    temb_dim=nf * 4,
                    spatial_ch=target_ch
                )
        # ============================================================

        # Build main U-Net (IDENTICAL to original)
        modules = []
        
        if embedding_type == 'fourier':
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, 
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4)
        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4)
        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2
                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), 
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                       num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, time_cond, class_logits=None, cross_attention=None):
        modules = self.all_modules
        m_idx = 0

        # Convert to 4-channel representation
        x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag,
                      x[:,[1],:,:].real, x[:,[1],:,:].imag), dim=1)

        # Time embedding
        if self.embedding_type == 'fourier':
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1
        elif self.embedding_type == 'positional':
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)
        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if not self.centered:
            x = 2 * x - 1.

        # Encode logits with uncertainty modeling
        logits_encoded = None
        uncertainty = None
        if class_logits is not None:
            B = x.shape[0]
            # Handle different input formats
            if class_logits.dim() == 2:
                class_logits = class_logits.unsqueeze(0)
            if class_logits.size(0) == 1 and B > 1:
                class_logits = class_logits.expand(B, -1, -1)
            
            # Hierarchical encoding with uncertainty
            logits_encoded, uncertainty = self.logits_encoder(class_logits)  # [B, S, D], [B, S, 1]

        # Downsampling
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        hs = [modules[m_idx](x)]
        m_idx += 1

        # Track scale for conditioning
        current_scale = 0
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                
                if h.shape[-2] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                
                # Apply advanced fusion at this scale
                if (logits_encoded is not None and 
                    current_scale in self.logits_cond_scales and
                    f'scale_{current_scale}' in self.fusion_blocks):
                    
                    # Multi-mechanism fusion
                    fused_features, alignment_weights = self.fusion_blocks[f'scale_{current_scale}'](
                        h, logits_encoded, uncertainty
                    )
                    
                    # Context-aware gating
                    gated_features = self.context_gates[f'scale_{current_scale}'](
                        temb, h, fused_features, uncertainty
                    )
                    
                    # Add to features
                    h = h + gated_features
                
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1
                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                
                hs.append(h)
                current_scale += 1

        # Bottleneck
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        # Apply conditioning at bottleneck
        if (logits_encoded is not None and 
            current_scale in self.logits_cond_scales and
            f'scale_{current_scale}' in self.fusion_blocks):
            
            fused_features, _ = self.fusion_blocks[f'scale_{current_scale}'](
                h, logits_encoded, uncertainty
            )
            
            gated_features = self.context_gates[f'scale_{current_scale}'](
                temb, h, fused_features, uncertainty
            )
            
            h = h + gated_features

        pyramid = None

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)
                        pyramid_h = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)

        # Convert back to complex
        h = self.output_layer(h)

        if self.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        
        return h