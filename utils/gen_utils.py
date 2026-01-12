import torch
from torch import Tensor
from tqdm import tqdm
from diffusers import DDPMScheduler
from custom_diffusers.continuous_ddpm import ContinuousDDPM
from einops import reduce

@torch.inference_mode()
def generate_samples2(
    clean_samples: Tensor,
    cond_map: Tensor,
    scheduler: DDPMScheduler,
    sample_steps: int,
    model: torch.nn.Module,
    disable=False,
):
    """Generate samples from a trained model (optimized)."""

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # Use only the first channel but avoid unnecessary copies
    # clean_samples: (B, C, T, H, W) -> (B, 1, T, H, W)
    clean_samples = clean_samples[:, 0:1, ...]  # slice keeps view if possible

    # Cond map to device once
    cond_map = cond_map.to(device=device, dtype=dtype, non_blocking=True)


    # Start from pure noise
    gen_sample = torch.randn_like(clean_samples, device=device, dtype=dtype)

    # IMPORTANT: timesteps are assumed to be set OUTSIDE this function
    timesteps = scheduler.timesteps.to(device)

    # Precompute continuous timesteps ONLY ONCE if needed
    if isinstance(scheduler, ContinuousDDPM):
        # steps has length sample_steps + 1, but we only index by timesteps
        steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        # log_snr for all timesteps in one go
        t_all = scheduler.log_snr(steps[timesteps])  # shape: (n_steps,)

    batch_size = clean_samples.shape[0]

    for step_idx, t_idx in tqdm(
        enumerate(timesteps),
        total=len(timesteps),
        desc="Sampling",
        disable=disable,
    ):
        if isinstance(scheduler, ContinuousDDPM):
            # Repeat scalar for batch
            t = t_all[step_idx].expand(batch_size)
        else:
            t = t_idx

        # Model forward
        output = model(
            gen_sample,
            t,
            cond_map=cond_map,
        )

        # One reverse diffusion step
        gen_sample = scheduler.step(
            output,
            timestep=t_idx,
            sample=gen_sample
        ).prev_sample

    return gen_sample

@torch.inference_mode()
def generate_samples_fcn3(
    cond_map: Tensor,              # [B, Ccond, H, W] or [B, Ccond, 1, H, W]
    model: torch.nn.Module,
    noise_channels: int,
    ensemble_size: int = 1,
):
    """
    FCN3-style sampling: one forward pass per ensemble member.
    Returns:
      if ensemble_size==1: [B, 1, 1, H, W]
      else:               [E, B, 1, 1, H, W]
    """
    print('call fcn3')
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    c = cond_map.to(device=device, dtype=dtype, non_blocking=True)

    # ensure cond is 5D: [B, Ccond, 1, H, W]
    if c.ndim == 4:
        c = c.unsqueeze(2)
    elif c.ndim == 6 and c.shape[3] == 1:
        c = c.squeeze(3)
    assert c.ndim == 5, f"cond_map must be 5D after fix, got {c.shape}"

    B, Cc, F, H, W = c.shape
    assert F == 1, f"Expected F=1 for annual FCN3, got F={F}"

    # dummy timestep required by your model signature
    t = torch.zeros(B, device=device, dtype=torch.long)

    def one():
        x = torch.randn(B, noise_channels, 1, H, W, device=device, dtype=dtype)
        return model(x, t, cond_map=c)  # -> [B, V, 1, H, W] (V=1 for now)

    if ensemble_size == 1:
        return one()
    else:
        outs = [one() for _ in range(ensemble_size)]
        return torch.stack(outs, dim=0)

@torch.inference_mode()
def generate_samples(
    clean_samples: Tensor,
    cond_map: Tensor,
    scheduler: DDPMScheduler,
    sample_steps: int,
    model: torch.nn.Module,
    disable=False,
):
    """Generate samples from a trained model"""
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype


    # Average across the time dimension, and then repeat along the time dimension
    # To get our average monthly conditioning map
    #cond_map = reduce(clean_samples, "b v t h w -> b v 1 h w", "mean").repeat(
    #    1, 1, clean_samples.shape[-3], 1, 1
    #)

    # Sample noise that we'll add to the clean images
    print(cond_map.shape,"cond map shape")
    clean_samples= clean_samples[:,0,:,:,:].unsqueeze(1)
    gen_sample = torch.randn_like(clean_samples)
    gen_samples = gen_sample.to(device=device, dtype=dtype)
    cond_map      = cond_map.to(device=device, dtype=dtype)
    #print(gen_sample.shape)
    #print(cond_map.shape)
    # set step values
    scheduler.set_timesteps(sample_steps)

    # Run the diffusion process in reverse
    for i in tqdm(
        scheduler.timesteps,
        "Sampling",
        disable=disable,
    ):
        # If we are using a continuous scheduler, convert the timestep to a log_snr
        if isinstance(scheduler, ContinuousDDPM):
            steps = torch.linspace(1.0, 0.0, sample_steps + 1, device=gen_sample.device)
            t = scheduler.log_snr(steps[i]).repeat(clean_samples.shape[0])
        else:
            t = i
        #print(i)
        output = model(
            gen_sample,
            t,
            cond_map=cond_map,
        )

        gen_sample = scheduler.step(output, timestep=i, sample=gen_sample).prev_sample

    return gen_sample
