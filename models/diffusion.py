
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your 3D model architecture (you uploaded video_net.py separately)
from video_net import UNetModel3D

# -----------------------------
# Helpers
# -----------------------------

def timestep_embedding(t, dim):
    device = t.device
    half = dim // 2
    if half < 1:
        return t.float().unsqueeze(1)
    freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1 if half > 1 else 1)))
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# -----------------------------
# 2D-facing wrapper around your 3D UNet
# Keeps train.py API the same: forward(x, cond, t) -> (B,1,H,W)
# Internally: adds a singleton time dim (T=1) and calls UNetModel3D
# -----------------------------


class UNet(nn.Module):
    """
    Compatibility wrapper so build_model_from_config(**cfg) with keys like
    in_channels/out_channels/base_ch/ch_mults/groups works with UNetModel3D.
    We assume a single target variable (out_channels == 1) and pass the
    conditioning map separately (cond_map) each forward.
    """
    def __init__(
        self,
        in_channels: int = 2,     # (ignored, UNetModel3D derives in/out from n_vars & cond_map)
        out_channels: int = 1,    # number of target variables (n_vars)
        base_ch: int = 64,        # -> model_dim
        ch_mults=(1, 2, 4),       # -> dim_mults
        num_res_blocks: int = 2,  # (unused; UNetModel3D fixes 2 per level internally)
        time_dim: int = 256,      # (unused; UNetModel3D derives)
        groups: int = 8,          # -> resnet_groups
        dropout: float = 0.0,     # (unused)
        attn_heads: int = 8,
        attn_dim_head: int = 32,
        use_sparse_linear_attn: bool = True,
        use_mid_attn: bool = False,
        init_kernel_size: int = 7,
        use_checkpoint: bool = False,
        use_temp_attn: bool = True,
        day_cond: bool = False,
        year_cond: bool = False,
        cond_map: bool = True,
    ):
        super().__init__()
        # n_vars = number of predicted channels
        n_vars = out_channels
        self.net = UNetModel3D(
            n_vars=n_vars,
            model_dim=base_ch,
            dim_mults=tuple(ch_mults),
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head,
            use_sparse_linear_attn=use_sparse_linear_attn,
            use_mid_attn=use_mid_attn,
            init_kernel_size=init_kernel_size,
            resnet_groups=groups,
            use_checkpoint=use_checkpoint,
            use_temp_attn=use_temp_attn,
            day_cond=day_cond,
            year_cond=year_cond,
            cond_map=cond_map,
        )

    # model.UNet.forward
    def forward(self, x_t, cond, t):
        # x_t: [B,1,H,W] or [B,1,F,H,W]   (target can be single-frame)
        # cond: [B,1,H,W] or [B,1,F,H,W]  (we want F=K here!)
        if x_t.ndim == 4:         # add F only for targets
            x_t = x_t.unsqueeze(2)      # -> [B,1,1,H,W]
        if cond is not None and cond.ndim == 4:
            # if cond already 5D, leave it; if 4D, add F=1
            cond = cond.unsqueeze(2)     # -> [B,1,1,H,W]
        out = self.net(x_t, t, days=None, years=None, cond_map=cond)
        if out.ndim == 5 and out.shape[2] == 1:
            out = out.squeeze(2)
        return out

# -----------------------------
# Diffusion wrapper (2D API)
# Uses standard DDPM math in 2D, but calls the wrapped 3D model under the hood.
# -----------------------------

class Diffusion(nn.Module):
    def __init__(self, model, img_channels=1, timesteps=1000, beta_schedule="linear"):
        super().__init__()
        self.model = model
        self.img_channels = img_channels
        self.T = timesteps

        if beta_schedule == "linear":
            beta_start, beta_end = 1e-4, 2e-2
            betas = torch.linspace(beta_start, beta_end, timesteps)
        else:
            raise ValueError("Only 'linear' beta_schedule implemented")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

    @torch.no_grad()
    def p_sample(self, x_t, cond, t):
        # Standard DDPM update but noise prediction comes from the wrapped 3D model
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)

        eps_theta = self.model(x_t, cond, t)  # -> (B,1,H,W)
        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)

        if (t == 0).all():
            return model_mean
        else:
            posterior_var_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, cond, shape, device):
        # shape: (B,1,H,W) — same as before
        B, _, H, W = shape
        x = torch.randn(shape, device=device)
        for tt in reversed(range(self.T)):
            t_tensor = torch.full((B,), tt, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, t_tensor)
        return x

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_a_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a_bar * x0 + sqrt_one_minus_a_bar * noise, noise

    def loss(self, x0, cond):
        B = x0.size(0)
        t = torch.randint(0, self.T, (B,), device=x0.device).long()
        x_t, noise = self.q_sample(x0, t)
        eps_pred = self.model(x_t, cond, t)
        return F.mse_loss(eps_pred, noise)
