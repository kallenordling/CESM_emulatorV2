# Standard library imports
import os
from typing import Union
from collections import OrderedDict
from hydra.utils import instantiate
# Third party imports
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import hydra
from omegaconf.omegaconf import DictConfig
from hydra.utils import instantiate
from accelerate import Accelerator
from diffusers import DDPMScheduler
import xarray as xr
from tqdm import tqdm
import pandas as pd
from custom_diffusers.continuous_ddpm import ContinuousDDPM
# Local imports
from data.climate_dataset import ClimateDataset
from utils.gen_utils import generate_samples,generate_samples2
from omegaconf import OmegaConf
from models.video_net import UNetModel3D
from ema_pytorch import EMA
Checkpoint = dict[str, Union[int, OrderedDict]]

# Assumes that gen is conditioned on val, and the first realization
# is always reserved for the test set
realization_dict = {"gen": "r10i1181p1f1", "val": "r10i1181p1f1", "test": "r10i1181p1f1"}


def get_starting_index(directory: str) -> int:
    """Goes through a directory of files named "member_i.nc" and returns the next available index."""
    files = os.listdir(directory)
    indices = [
        int(file.split("_")[1].split(".")[0])
        for file in files
        if file.startswith("member")
    ]
    return max(indices) + 1 if indices else 0


def create_batches(
    xr_ds: xr.Dataset,
    dataset: ClimateDataset,
) -> list[xr.Dataset]:
    """Splits the dataset up into batches of size batch_size. This is helpful
    for when we perform multiprocessing, so we can distribute the batches to different
    processes.

    Args:
        xr_ds (xr.Dataset): The xarray dataset that we want to split
        batch_size (int): The size of each batch
        gpu_ids (list[int]): A list of the GPUs we will be utilizing

    Returns:
        list[list[xr.Dataset]]: A list of all the batches, where each batch is a list of xarray datasets
    """
    seq_len = dataset.seq_len

    # Store a list of all batches, and a single batch
    data = []

    # Iterate through every 28 days in the xr dataset
    for i in range(0, len(xr_ds.year), seq_len):
        # Grab a single batch and convert it to tensor
        batch = xr_ds.isel(year=slice(i, i + seq_len))
        tensor_data = dataset.convert_xarray_to_tensor(batch)

        # Append the batch to the list of batches
        data.append((tensor_data, dict(batch.coords)))

    return data


def custom_collate_fn(
    batches: list[tuple[Tensor, xr.DataArray]],
) -> tuple[Tensor, list[xr.DataArray]]:
    """Collate function for the dataloader. This is necessary because we want to keep track of the time coordinates
    for each batch, so we can convert the generated tensors back into xarray datasets

    Args:
        batches (list[tuple[Tensor, xr.DataArray]]): A list of tuples, where each tuple contains a batch of tensors
        and the corresponding time coordinates

    Returns:
        tuple[Tensor, list[xr.DataArray]]: A tuple containing the stacked tensor batch and a list of coordinates
    """
    tensor_batch = []
    coords = []
    for batch in batches:
        tensor_batch.append(batch[0])
        coords.append(batch[1])

    return torch.stack(tensor_batch), coords


@hydra.main(version_base=None, config_path="./configs", config_name="generate_aero_ssp370")
def main(config: DictConfig) -> None:
    # Verify that the save folder exists
    assert os.path.isdir(config.save_dir), "Save directory does not exist"
    assert config.gen_mode in ["gen", "val", "test"], "Invalid gen mode"

    # If we're generating, make sure we have a load path
    if config.gen_mode == "gen":
        assert config.load_path, "Must specify a load path"
        assert os.path.isfile(config.load_path), "Invalid load path"

    # Make sure num samples is 1 if gen mode is not gen
    assert (
        config.samples_per == 1 or config.gen_mode == "gen"
    ), "Number of samples must be 1 for val and test"

    # Initialize all necessary objects
    accelerator = Accelerator(**config.accelerator)

    dataset: ClimateDataset = instantiate(
        config.dataset,
        data_dir=config.data_dir,
        realizations=[realization_dict[config.gen_mode]],
        target_vars=config.variables,cond_vars=["CO2_em_anthro",'sul'],cond_file=config.cond_file

    )
    scheduler: ContinuousDDPM = instantiate(config.scheduler)
    scheduler.set_timesteps(config.sample_steps)

    conf = OmegaConf.load('./configs/config_aero.yaml')
    model_conf: UNetModel3D = instantiate(conf.model)

    if config.gen_mode == "gen":
        # Load the model from the checkpoint
        chkpt: Checkpoint = torch.load(config.load_path, map_location="cpu", weights_only=False)
        #model = chkpt["EMA"].eval()
        #model = model.to(accelerator.device)
        
        ema_model_sd = chkpt["EMA"]  # full EMA state dict (online_model + ema_model)

        # Extract only EMA weights and strip "ema_model." prefix
        #ema_model_sd = {
        #  k.replace("ema_model.", ""): v
        #  for k, v in ema_wrapped_sd.items()
        #  if k.startswith("ema_model.")
        #}

        
        model = EMA(
            model_conf,
            beta=0.9999,  # exponential moving average factor
            update_after_step=100,  # only after this number of .update() calls will it start updating
            update_every=10,
        ).to(accelerator.device)
        model.ema_model.load_state_dict(ema_model_sd)
        model.eval()

    else:
        model = None

    # Grab the Xarray dataset from the dataset object
    xr_ds = dataset.dataset_cond.load()
    print(xr_ds)
    # Restrict days to the first 28 days of each month and select years
    #xr_ds = xr_ds.sel(time=xr_ds.time.dt.day.isin(range(1, 29)))
    xr_ds = xr_ds.sel(year=slice(str(config.start_year), str(config.end_year)))
    print(xr_ds)
    input('wait')
    batches = create_batches(xr_ds, dataset)

    dataloader = DataLoader(
        batches, batch_size=config.batch_size, collate_fn=custom_collate_fn
    )

    # Prepare the model and dataloader for distributed training
    model, dataloader = accelerator.prepare(model, dataloader)

    for i in tqdm(range(config.samples_per)):
        gen_samples = []

        for tensor_batch, coords in tqdm(
            dataloader, disable=not accelerator.is_main_process
        ):
            #print(i, "### I ####")
            #print(tensor_batch.shape)
            #print(coords)
            tensor_batch = tensor_batch.to(accelerator.device)
            if model is not None:
                gen_months = generate_samples2(
                    tensor_batch,tensor_batch,
                    scheduler=scheduler,
                    sample_steps=config.sample_steps,
                    model=model,
                    disable=True,
                )
            else:
                gen_months = tensor_batch

            for i in range(len(gen_months)):
                gen_samples.append(
                    dataset.convert_tensor_to_xarray(gen_months[i], coords=coords[i])
                )

        gen_samples = accelerator.gather_for_metrics(gen_samples)
        gen_samples = xr.concat(gen_samples, "year").sortby("year")

        if accelerator.is_main_process:

            # If we are generating multiple samples, create a directory for them
            save_name = f"{config.gen_mode}_{config.save_name + '_' if config.save_name is not None else ''}{'_'.join(config.variables)}_{config.start_year}-{config.end_year}.nc"
            save_path = os.path.join(
                config.data_dir, save_name
            )
            if config.gen_mode == "gen" and config.samples_per > 1:
                save_dir = save_path.strip(".nc")
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)

                mem_index = get_starting_index(save_dir)
                save_path = os.path.join(save_dir, f"member_{mem_index}.nc")

            else:
                # Delete the file if it already exists (avoids permission denied errors)
                if os.path.isfile(save_path):
                    os.remove(save_path)

            # Save the generated samples
            print('save file',save_path)
            gen_samples.to_netcdf(save_path)

            os.chmod(save_path, 0o770)


if __name__ == "__main__":
    main()
