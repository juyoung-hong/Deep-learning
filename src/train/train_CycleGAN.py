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
import itertools

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
        'Train/GAN_Loss': MeanMetric().to(device),
        'Train/Identity_Loss': MeanMetric().to(device),
        'Train/Cycle_Loss': MeanMetric().to(device),
        'Train/G_Loss': MeanMetric().to(device),
        'Train/D_A_Loss': MeanMetric().to(device),
        'Train/D_B_Loss': MeanMetric().to(device),
        'Train/D_Loss': MeanMetric().to(device),
        'Test/GAN_Loss': MeanMetric().to(device),
        'Test/Identity_Loss': MeanMetric().to(device),
        'Test/Cycle_Loss': MeanMetric().to(device),
        'Test/G_Loss': MeanMetric().to(device),
        'Test/D_A_Loss': MeanMetric().to(device),
        'Test/D_B_Loss': MeanMetric().to(device),
        'Test/D_Loss': MeanMetric().to(device),
    }

    return metrics

def save_figures(real_A, real_B, fake_A, fake_B, cfg, epoch, logger, test):
    prefix = ('best' if test else 'epoch')

    real_A = make_grid(real_A, nrow=cfg.data.batch_size, normalize=True, value_range=(-1, 1))
    real_B = make_grid(real_B, nrow=cfg.data.batch_size, normalize=True, value_range=(-1, 1))
    fake_A = make_grid(fake_A, nrow=cfg.data.batch_size, normalize=True, value_range=(-1, 1))
    fake_B = make_grid(fake_B, nrow=cfg.data.batch_size, normalize=True, value_range=(-1, 1))
    grid_image = torch.cat((real_A, fake_B, real_B, fake_A), 1)

    save_fake_path = Path(cfg.trainer.save_ckpt_dirpath) / (
        cfg.trainer.save_generated_filename + f"/fake_{prefix}_{epoch}.png"
    )
    save_fake_path.parent.mkdir(parents=True, exist_ok=True)
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
    test_dataloader = datamodule.test_dataloader()

    device = cfg.trainer.accelerator
    log.info(f"Instantiating model <{cfg.model._target_}>")
    modelA = hydra.utils.instantiate(cfg.model).to(device)
    modelB = hydra.utils.instantiate(cfg.model).to(device)
    G_AB = modelA.get_generator()
    G_BA = modelB.get_generator()
    D_A = modelA.get_discriminator()
    D_B = modelB.get_discriminator()

    # Get Loss function
    gan_criterion = hydra.utils.instantiate(cfg.gan_loss)()
    cycle_criterion = hydra.utils.instantiate(cfg.cycle_loss)()
    identity_criterion = hydra.utils.instantiate(cfg.identity_loss)()

    # Get Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer_G = optimizer(params=itertools.chain(G_AB.parameters(), G_BA.parameters()))
    
    optimizer_D_A = optimizer(params=D_A.parameters())
    optimizer_D_B = optimizer(params=D_B.parameters())

    # Get Metrics
    metrics = get_metrics(device)

    log.info(f"Instantiating logger <{cfg.logger._target_}>")
    logger = hydra.utils.instantiate(cfg.get("logger"))

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": modelA,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict, log)

    if cfg.get("compile"):
        log.info("Compiling model!")
        modelA = torch.compile(modelA)
        modelB = torch.compile(modelB)

    if cfg.get("train"):
        log.info("Starting training!")
        for epoch in range(cfg.trainer.epochs):
            # train step start
            G_AB.train()
            G_BA.train()
            D_A.train()
            D_B.train()
            for k, v in metrics.items():
                if k.startswith('Train/'):
                    v.reset()

            for step, images in enumerate(tqdm(train_dataloader)):
                global_step = epoch * len(train_dataloader) + step

                real_A = images['A'].to(device)
                real_B = images['B'].to(device)
                # ---------------------
                #  Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                # ---------------------
                # train D_A
                optimizer_D_A.zero_grad()
                D_A.zero_grad()

                outputs = D_A(real_A)
                labels = torch.ones_like(outputs, device=device)
                real_d_loss = gan_criterion(outputs, labels)
                real_d_loss.backward()

                fake_A = G_BA(real_B)
                outputs = D_A(fake_A.detach())
                labels = torch.zeros_like(outputs, device=device)
                fake_d_loss = gan_criterion(outputs, labels)
                fake_d_loss.backward()

                d_a_loss = (real_d_loss + fake_d_loss) / 2
                optimizer_D_A.step()

                # train D_B
                optimizer_D_B.zero_grad()
                D_B.zero_grad()

                outputs = D_B(real_B)
                labels = torch.ones_like(outputs, device=device)
                real_d_loss = gan_criterion(outputs, labels)
                real_d_loss.backward()

                fake_B = G_AB(real_A)
                outputs = D_B(fake_B.detach())
                labels = torch.zeros_like(outputs, device=device)
                fake_d_loss = gan_criterion(outputs, labels)
                fake_d_loss.backward()

                d_b_loss = (real_d_loss + fake_d_loss) / 2
                optimizer_D_B.step()

                d_loss = d_a_loss + d_b_loss
                # -----------------
                #  Train Generator: maximize log(D(G(z)))
                # -----------------
                optimizer_G.zero_grad()
                G_AB.zero_grad()
                G_BA.zero_grad()

                # identity loss
                identity_a_loss = identity_criterion(G_BA(real_A), real_A)
                identity_b_loss = identity_criterion(G_AB(real_B), real_B)
                identity_loss = (identity_a_loss + identity_b_loss) / 2
                
                # gan loss
                labels = torch.ones_like(outputs, device=device)
                gan_ab_loss = gan_criterion(D_B(fake_B), labels)
                gan_ba_loss = gan_criterion(D_A(fake_A), labels)
                gan_loss = (gan_ab_loss + gan_ba_loss) / 2

                # cycle loss
                reconstructed_A = G_BA(fake_B)
                reconstructed_B = G_AB(fake_A)
                cycle_a_loss = cycle_criterion(reconstructed_A, real_A)
                cycle_b_loss = cycle_criterion(reconstructed_B, real_B)
                cycle_loss = (cycle_a_loss + cycle_b_loss) / 2
                
                g_loss = gan_loss + cfg.lambda_cycle*cycle_loss + cfg.lambda_identity*identity_loss
                g_loss.backward()
                optimizer_G.step()

                # update metrics
                metrics['Train/GAN_Loss'](gan_loss)
                metrics['Train/Identity_Loss'](identity_loss)
                metrics['Train/Cycle_Loss'](cycle_loss)
                metrics['Train/G_Loss'](g_loss)
                metrics['Train/D_A_Loss'](d_a_loss)
                metrics['Train/D_B_Loss'](d_b_loss)
                metrics['Train/D_Loss'](d_loss)

                # Get metric
                if step % cfg.trainer.log_every_n_step == 0:
                    for k, v in metrics.items():
                        if k.startswith('Train/'):
                            train_loss = v.compute()
                            message = f"Epoch: {epoch} {k}: {train_loss}"
                            log.info(message)
                            logger.add_scalar(k, train_loss, global_step)
                
            if epoch % cfg.trainer.save_ckpt_every_n_epoch == 0:
                save_path = Path(cfg.trainer.save_ckpt_dirpath) / (
                    cfg.trainer.save_filename + f"/epoch_{epoch}.pth"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "epoch": epoch,
                    "modelA": modelA.state_dict(),
                    "modelB": modelB.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D_A": optimizer_D_A.state_dict(),
                    "optimizer_D_B": optimizer_D_B.state_dict(),
                }
                torch.save(state, save_path)

            if epoch % cfg.trainer.generate_every_n_epoch == 0:
                save_figures(real_A, real_B, fake_A, fake_B, cfg, epoch, logger, False)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.trainer.get("best_model_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        # ---------------------
        #  Test
        # ---------------------   
        G_AB.eval()
        G_BA.eval()
        D_A.eval()
        D_B.eval()
        for k, v in metrics.items():
            if k.startswith('Test/'):
                v.reset()

        for step, images in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                real_A = images['A'].to(device)
                real_B = images['B'].to(device)

                outputs = D_A(real_A)
                labels = torch.ones_like(outputs, device=device)
                real_d_loss = gan_criterion(outputs, labels)

                fake_A = G_BA(real_B)
                outputs = D_A(fake_A)
                labels = torch.zeros_like(outputs, device=device)
                fake_d_loss = gan_criterion(outputs, labels)

                d_a_loss = (real_d_loss + fake_d_loss) / 2

                labels = torch.ones_like(outputs, device=device)
                outputs = D_B(real_B)
                real_d_loss = gan_criterion(outputs, labels)

                fake_B = G_AB(real_A)
                outputs = D_B(fake_B)
                labels = torch.zeros_like(outputs, device=device)
                fake_d_loss = gan_criterion(outputs, labels)

                d_b_loss = (real_d_loss + fake_d_loss) / 2

                d_loss = d_a_loss + d_b_loss

                # identity loss
                identity_a_loss = identity_criterion(G_BA(real_A), real_A)
                identity_b_loss = identity_criterion(G_AB(real_B), real_B)
                identity_loss = (identity_a_loss + identity_b_loss) / 2
                
                # gan loss
                labels = torch.ones_like(outputs, device=device)
                gan_ab_loss = gan_criterion(D_B(fake_B), labels)
                gan_ba_loss = gan_criterion(D_A(fake_A), labels)
                gan_loss = (gan_ab_loss + gan_ba_loss) / 2

                # cycle loss
                reconstructed_A = G_BA(fake_B)
                reconstructed_B = G_AB(fake_A)
                cycle_a_loss = cycle_criterion(reconstructed_A, real_A)
                cycle_b_loss = cycle_criterion(reconstructed_B, real_B)
                cycle_loss = (cycle_a_loss + cycle_b_loss) / 2
                
                g_loss = gan_loss + cfg.lambda_cycle*cycle_loss + cfg.lambda_identity*identity_loss

            # update metrics
            metrics['Test/GAN_Loss'](gan_loss)
            metrics['Test/Identity_Loss'](identity_loss)
            metrics['Test/Cycle_Loss'](cycle_loss)
            metrics['Test/G_Loss'](g_loss)
            metrics['Test/D_A_Loss'](d_a_loss)
            metrics['Test/D_B_Loss'](d_b_loss)
            metrics['Test/D_Loss'](d_loss)
        
        for k, v in metrics.items():
            if k.startswith('Test/'):
                test_loss = v.compute()
                message = f"{k}: {test_loss}"
                log.info(message)
                logger.add_scalar(k, test_loss, epoch)

        save_figures(real_A, real_B, fake_A, fake_B, cfg, epoch, logger, True)


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
