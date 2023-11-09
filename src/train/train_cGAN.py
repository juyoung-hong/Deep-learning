# References
# https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from torch.autograd import Variable
from torchmetrics import MeanMetric, MinMetric
from torchvision.utils import make_grid, save_image

# from torchmetrics.image.inception import InceptionScore
# from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #

from src import utils

log = utils.get_pylogger(__name__)

def get_metrics(device):

    metrics = {
        'Train/G_Loss': MeanMetric().to(device),
        'Train/D_Loss': MeanMetric().to(device),
        'Val/G_Loss': MeanMetric().to(device),
        'Val/D_Loss': MeanMetric().to(device),
        'Test/G_Loss': MeanMetric().to(device),
        'Test/D_Loss': MeanMetric().to(device),
    }

    return metrics

def save_figures(real_images, fake_images, cfg, epoch, logger, save_real, test):
    prefix = ('best' if test else 'epoch')
    if save_real:
        filename = ('real_epoch' if epoch == 0 else "real_best")
        save_real_path = Path(cfg.trainer.save_ckpt_dirpath) / (
        cfg.trainer.save_generated_filename + f"/{filename}.png"
        )
        save_real_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image = make_grid(
        real_images, nrow=3, normalize=True, value_range=(-1, 1)
        )
        save_image(grid_image, save_real_path)

    save_fake_path = Path(cfg.trainer.save_ckpt_dirpath) / (
        cfg.trainer.save_generated_filename + f"/fake_{prefix}_{epoch}.png"
    )
    save_fake_path.parent.mkdir(parents=True, exist_ok=True)
    grid_image = make_grid(
        fake_images, nrow=3, normalize=True, value_range=(-1, 1)
    )
    save_image(grid_image, save_fake_path)
    if not test:
        logger.add_image("image", grid_image, epoch)

def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        worker_init_fn = utils.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data, worker_init_fn=worker_init_fn)

    # Download data if needed.
    datamodule.prepare_data()

    # Load data.
    datamodule.setup()

    # Get data loader.
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    device = cfg.trainer.accelerator
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)
    G = model.get_generator()
    D = model.get_discriminator()
    G.device = device

    # Get Loss function
    criterion = hydra.utils.instantiate(cfg.loss)()

    # Get Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer_G = optimizer(params=G.parameters())
    optimizer_D = optimizer(params=D.parameters())

    # Get Metrics
    metrics = get_metrics(device)

    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict, log)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("train"):
        log.info("Starting training!")

        # Adversarial ground truths
        real_label = 1.0
        fake_label = 0.0

        for epoch in range(cfg.trainer.epochs):
            # train step start
            G.train()
            D.train()
            for k, v in metrics.items():
                if k.startswith('Train/'):
                    v.reset()

            for step, (images, labels) in enumerate(tqdm(train_dataloader)):
                global_step = epoch * len(train_dataloader) + step
                # ---------------------
                #  Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                # ---------------------
                optimizer_D.zero_grad()
                D.zero_grad()
                real_images = images.to(device)
                labels = labels.to(device)
                gan_labels = torch.full(
                    (cfg.data.batch_size,), real_label, dtype=torch.float, device=device
                )
                outputs = D(real_images, labels).view(-1)
                real_d_loss = criterion(outputs, gan_labels)
                real_d_loss.backward()

                fake_images = G(cfg.data.batch_size, labels)
                gan_labels.fill_(fake_label)
                outputs = D(fake_images.detach(), labels).view(-1)
                fake_d_loss = criterion(outputs, gan_labels)
                fake_d_loss.backward()

                d_loss = real_d_loss + fake_d_loss
                optimizer_D.step()

                # -----------------
                #  Train Generator: maximize log(D(G(z)))
                # -----------------
                optimizer_G.zero_grad()
                G.zero_grad()
                gan_labels.fill_(real_label)
                outputs = D(fake_images, labels).view(-1)
                g_loss = criterion(outputs, gan_labels)
                g_loss.backward()
                optimizer_G.step()

                # update metrics
                metrics['Train/G_Loss'](d_loss)
                metrics['Train/D_Loss'](g_loss)

                # Get metric
                if step % cfg.trainer.log_every_n_step == 0:
                    for k, v in metrics.items():
                        if k.startswith('Train/'):
                            train_loss = v.compute()
                            message = f"Epoch: {epoch} {k}: {train_loss}"
                            log.info(message)
                            logger.add_scalar(k, train_loss, global_step)

            if epoch % cfg.trainer.check_val_every_n_epoch == 0:
                # ---------------------
                #  Validation
                # --------------------- 
                G.eval()
                D.eval()
                for k, v in metrics.items():
                    if k.startswith('Val/'):
                        v.reset()

                for step, (images, labels) in enumerate(tqdm(val_dataloader)):
                    with torch.no_grad():
                        real_images = images.to(device)
                        labels = labels.to(device)
                        gan_labels = torch.full(
                            (cfg.data.batch_size,), real_label, dtype=torch.float, device=device
                        )
                        
                        outputs = D(real_images, labels).view(-1)
                        real_d_loss = criterion(outputs, gan_labels)

                        fake_images = G(cfg.data.batch_size, labels)
                        gan_labels.fill_(fake_label)
                        outputs = D(fake_images.detach(), labels).view(-1)
                        fake_d_loss = criterion(outputs, gan_labels)

                        d_loss = real_d_loss + fake_d_loss

                        gan_labels.fill_(real_label)
                        outputs = D(fake_images, labels).view(-1)
                        g_loss = criterion(outputs, gan_labels)

                    # update metrics
                    metrics['Val/G_Loss'](g_loss)
                    metrics['Val/D_Loss'](d_loss)
                
                for k, v in metrics.items():
                    if k.startswith('Val/'):
                        val_loss = v.compute()
                        message = f"Epoch: {epoch} {k}: {val_loss}"
                        log.info(message)
                        logger.add_scalar(k, val_loss, epoch)

            if epoch % cfg.trainer.save_ckpt_every_n_epoch == 0:
                save_path = Path(cfg.trainer.save_ckpt_dirpath) / (
                    cfg.trainer.save_filename + f"/epoch_{epoch}.pth"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                }
                torch.save(state, save_path)

            if epoch % cfg.trainer.generate_every_n_epoch == 0:
                real_images = images[:9]
                fake_images = fake_images[:9]
                save_real = (True if epoch == 0 else False)
                save_figures(real_images, fake_images, cfg, epoch, logger, save_real, False)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.trainer.get("best_model_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        # ---------------------
        #  Test
        # ---------------------   
        G.eval()
        D.eval()
        for k, v in metrics.items():
            if k.startswith('Test/'):
                v.reset()

        for step, (images, labels) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                real_images = images.to(device)
                labels = labels.to(device)
                gan_labels = torch.full(
                    (cfg.data.batch_size,), real_label, dtype=torch.float, device=device
                )
                
                outputs = D(real_images, labels).view(-1)
                real_d_loss = criterion(outputs, gan_labels)

                fake_images = G(cfg.data.batch_size, labels)
                gan_labels.fill_(fake_label)
                outputs = D(fake_images.detach(), labels).view(-1)
                fake_d_loss = criterion(outputs, gan_labels)

                d_loss = real_d_loss + fake_d_loss

                gan_labels.fill_(real_label)
                outputs = D(fake_images, labels).view(-1)
                g_loss = criterion(outputs, gan_labels)

            # update metrics
            metrics['Test/G_Loss'](g_loss)
            metrics['Test/D_Loss'](d_loss)
        
        for k, v in metrics.items():
            if k.startswith('Test/'):
                test_loss = v.compute()
                message = f"{k}: {test_loss}"
                log.info(message)
                logger.add_scalar(k, test_loss, epoch)

        real_images = images[:9]
        fake_images = fake_images[:9]
        save_figures(real_images, fake_images, cfg, epoch, logger, True, True)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg, log)

    # train the model
    try:
        train(cfg)
    # things to do if exception occurs
    except Exception as ex:
        # save exception to `.log` file
        log.exception("")

        # some hyperparameter combinations might be invalid or cause out-of-memory errors
        # so when using hparam search plugins like Optuna, you might want to disable
        # raising the below exception to avoid multirun failure
        raise ex
    finally:
        # display output dir path in terminal
        log.info(f"Output dir: {cfg.paths.output_dir}")

        # always close wandb run (even if exception occurs so multirun won't fail)
        if find_spec("wandb"):  # check if wandb is installed
            import wandb

            if wandb.run:
                log.info("Closing wandb!")
                wandb.finish()


if __name__ == "__main__":
    main()
