from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDPMScheduler

from data.climate_dataset import ClimateDataset
from trainer.unetTrainer import UNetTrainer
from models.video_net import UNetModel3D


@hydra.main(version_base=None, config_path="configs", config_name="config_aero.yaml")
def main(cfg: DictConfig) -> None:
    # Create accelerator object and set RNG seed
    accelerator = Accelerator(**cfg.accelerator)
    set_seed(cfg.seed)

    # Logger works with distributed processes
    logger = get_logger(__name__, log_level="INFO")

    # Init logger
    logger.info(f"Instantiating datasets <{cfg.data.train._target_}>")
    train_cfg =  dict(cfg.data.train)


    # Avoid race conditions when loading data
    with accelerator.main_process_first():
        train_set: ClimateDataset = instantiate(
            train_cfg, _recursive_=False
        )


    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: UNetModel3D = instantiate(cfg.model)
    logger.info(str(model))

    logger.info(f"Instantiating scheduler <{cfg.scheduler._target_}>")
    scheduler: DDPMScheduler = instantiate(cfg.scheduler)



    logger.info(f"Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: UNetTrainer = instantiate(
        cfg.trainer,
        train_set,
        model=model,
        accelerator=accelerator,
        scheduler=scheduler,
    )

    print("Starting training")
    trainer.train()


if __name__ == "__main__":
    main()
