# References
# https://github.com/ashleve/lightning-hydra-template/blob/main/src/train.py

from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
import pyrootutils
import torch
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
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
    val_dataloader = datamodule.val_dataloader()

    device = cfg.trainer.accelerator
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model).to(device)

    # Get Loss function
    criterion = hydra.utils.instantiate(cfg.loss)()

    # Get Optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer(params=model.parameters())

    # Get Learning rate scheduler
    if cfg.get("scheduler"):
        scheduler = hydra.utils.instantiate(cfg.scheduler)
        scheduler = scheduler(optimizer=optimizer)

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

        # metric objects for calculating and averaging accuracy across batches
        train_acc = Accuracy(task="multiclass", num_classes=10).to(device)
        val_acc = Accuracy(task="multiclass", num_classes=10).to(device)

        # for averaging loss across batches
        train_loss = MeanMetric().to(device)
        val_loss = MeanMetric().to(device)

        # for tracking best so far validation accuracy
        val_acc_best = MaxMetric().to(device)

        for epoch in range(cfg.trainer.epochs):
            # train step start
            train_loss.reset()
            train_acc.reset()
            model.train()
            for images, labels in tqdm(train_dataloader):
                # forward + backward + optimize
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Get metric
                preds = torch.argmax(outputs, dim=1)
                batch_train_loss = train_loss(loss)
                batch_train_acc = train_acc(preds, labels)
                log.info(f"train/loss: {batch_train_loss}")
                log.info(f"train/acc: {batch_train_acc}")
            # train step end
            logger.add_scalar("Train/Loss", train_loss.compute(), epoch)
            logger.add_scalar("Train/Acc", train_acc.compute(), epoch)

            if epoch % cfg.trainer.check_val_every_n_epoch == 0:
                # validation step start
                val_loss.reset()
                val_acc.reset()
                val_acc_best.reset()
                model.eval()
                for images, labels in tqdm(val_dataloader):
                    # forward
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        preds = torch.argmax(outputs, dim=1)
                    # Get metric
                    batch_val_loss = val_loss(loss)
                    batch_val_acc = val_acc(preds, labels)
                    log.info(f"val/loss: {batch_val_loss}")
                    log.info(f"val/acc: {batch_val_acc}")
                # validation step end
                acc = val_acc.compute()  # get current val acc
                val_acc_best(acc)  # update best so far val acc
                log.info(f"val/acc_best: {val_acc_best.compute()}")
                logger.add_scalar("Val/Loss", val_loss.compute(), epoch)
                logger.add_scalar("Val/Acc", val_acc.compute(), epoch)

            if epoch % cfg.trainer.save_ckpt_every_n_epoch == 0:
                save_path = Path(cfg.trainer.save_dirpath) / (
                    cfg.trainer.save_filename + f"/epoch_{epoch}.pth"
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (
                        scheduler.state_dict() if cfg.get("scheduler") else None
                    ),
                }
                torch.save(state, save_path)

    if cfg.get("test"):
        log.info("Starting testing!")
        test_dataloader = datamodule.test_dataloader()
        ckpt_path = cfg.trainer.get("best_model_path")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None

        test_acc = Accuracy(task="multiclass", num_classes=10).to(device)
        test_loss = MeanMetric().to(device)
        # test step start
        test_loss.reset()
        test_acc.reset()
        model.eval()
        for images, labels in tqdm(test_dataloader):
            # forward
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
            # Get metric
            batch_test_loss = test_loss(loss)
            batch_test_acc = test_acc(preds, labels)
            log.info(f"test/loss: {batch_test_loss}")
            log.info(f"test/acc: {batch_test_acc}")
        # test step end
        acc = test_acc.compute()  # get current test acc
        logger.add_scalar("Test/Loss", test_loss.compute(), epoch)
        logger.add_scalar("Test/Acc", test_acc.compute(), epoch)


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
