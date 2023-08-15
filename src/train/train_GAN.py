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

    device = cfg.trainer.accelerator
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)
    G = model.get_generator()
    D = model.get_discriminator()

    # Get Loss function
    criterion = hydra.utils.instantiate(cfg.loss)()

    # Get Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer_G = optimizer(params=G.parameters())
    optimizer_D = optimizer(params=D.parameters())

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

        # metric objects for calculating inception score across batches
        # val_is = InceptionScore().to(device)
        # val_fid = FrechetInceptionDistance().to(device)

        # for averaging loss across batches
        train_d_loss = MeanMetric().to(device)
        train_g_loss = MeanMetric().to(device)

        # for tracking best so far validation IS, FID
        # val_is_best = MinMetric().to(device)
        # val_fid_best = MinMetric().to(device)

        # Adversarial ground truths
        real_label = 1.0
        fake_label = 0.0

        for epoch in range(cfg.trainer.epochs):
            # train step start
            train_d_loss.reset()
            train_g_loss.reset()
            G.train()
            D.train()
            for images, labels in tqdm(train_dataloader):
                # ---------------------
                #  Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                # ---------------------
                optimizer_D.zero_grad()
                D.zero_grad()
                real_images = images.to(device)
                labels = torch.full(
                    (cfg.data.batch_size,), real_label, dtype=torch.float, device=device
                )
                outputs = D(real_images).view(-1)
                real_d_loss = criterion(outputs, labels)
                real_d_loss.backward()

                fake_images = G()
                labels.fill_(fake_label)
                outputs = D(fake_images.detach()).view(-1)
                fake_d_loss = criterion(outputs, labels)
                fake_d_loss.backward()

                d_loss = real_d_loss + fake_d_loss
                optimizer_D.step()

                # -----------------
                #  Train Generator: maximize log(D(G(z)))
                # -----------------
                optimizer_G.zero_grad()
                G.zero_grad()
                labels.fill_(real_label)
                outputs = D(fake_images).view(-1)
                g_loss = criterion(outputs, labels)
                g_loss.backward()
                optimizer_G.step()

                # Get metric
                batch_train_d_loss = train_d_loss(d_loss)
                batch_train_g_loss = train_g_loss(g_loss)
                log.info(f"train/D_loss: {batch_train_d_loss}")
                log.info(f"train/G_loss: {batch_train_g_loss}")

            # train step end
            logger.add_scalar("Train/D_Loss", train_d_loss.compute(), epoch)
            logger.add_scalar("Train/G_Loss", train_g_loss.compute(), epoch)

            if epoch % cfg.trainer.check_val_every_n_epoch == 0:
                # validation step start
                # val_is.reset()
                # val_fid.reset()
                fake_images = []
                G.eval()
                for i in range(9):
                    with torch.no_grad():
                        fake_image = G()
                    fake_images.append(fake_image.squeeze(0))

                # Get metric
                # batch_val_is = val_is(fake_images)
                # batch_val_fid = val_fid(fake_images)
                # log.info(f"val/InceptionScore: {batch_val_is}")
                # log.info(f"val/FrechetInceptionDistance: {batch_val_fid}")

                # validation step end
                # is_ = val_is.compute()  # get current val IS
                # fid = val_fid.compute()  # get current val FID
                # val_is_best(is_)  # update best so far val IS
                # val_fid_best(fid)  # update best so far val FID
                # log.info(f"val/IS_best: {val_is_best.compute()}")
                # log.info(f"val/FID_best: {val_fid_best.compute()}")
                # logger.add_scalar("Val/IS", val_is.compute(), epoch)
                # logger.add_scalar("Val/FID", val_fid.compute(), epoch)

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
                save_path = Path(cfg.trainer.save_ckpt_dirpath) / (
                    cfg.trainer.save_generated_filename + f"/epoch_{epoch}.png"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                grid_image = make_grid(
                    fake_images, nrow=3, normalize=True, value_range=(-1, 1)
                )
                save_image(grid_image, save_path)
                logger.add_image("image", grid_image, epoch)

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = cfg.trainer.get("best_model_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        # test_is = InceptionScore().to(device)
        # test_fid = FrechetInceptionDistance().to(device)

        # test step start
        # test_is.reset()
        # test_fid.reset()

        fake_images = []
        G.eval()
        for i in range(9):
            with torch.no_grad():
                fake_image = G()
            fake_images.append(fake_image.squeeze(0))

        # Get metric
        # batch_test_is = test_is(fake_images)
        # batch_test_fid = test_fid(fake_images)
        # log.info(f"test/InceptionScore: {batch_test_is}")
        # log.info(f"test/FrechetInceptionDistance: {batch_test_fid}")

        # test step end
        # is_ = test_is.compute()  # get current test IS.
        # fid = test_fid.compute()  # get current test FID.
        # logger.add_scalar("Test/IS", test_is.compute(), epoch)
        # logger.add_scalar("Test/FID", test_fid.compute(), epoch)

        save_path = Path(cfg.trainer.save_ckpt_dirpath) / (
            cfg.trainer.save_generated_filename + f"/best.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image = make_grid(fake_images, nrow=3, normalize=True, value_range=(-1, 1))
        save_image(grid_image, save_path)
        logger.add_image("image", grid_image, epoch)


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
