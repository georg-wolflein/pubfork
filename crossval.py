from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Any, Iterator
import sys
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from omegaconf import DictConfig, ListConfig, OmegaConf
import hydra
import os
import numpy.typing as npt
import numpy as np
from sklearn.model_selection import KFold
from textwrap import indent

os.environ["HYDRA_FULL_ERROR"] = "1"

from pubfork.utils import (
    flatten_batched_dicts,
    make_preds_df,
)
from pubfork.data import BagDataset
from pubfork.model import LitMilClassificationMixin
from pubfork.targets import TargetEncoder
from train import train, load_dataset_df, summarize_dataset


def get_splits(
    items: npt.NDArray[Any], n_splits: int = 6
) -> Iterator[Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]]:
    """Splits a dataset into six training, validation and test sets

    This generator will yield `n_split` sets of training, validation and test
    sets.  To do so, it first splits `items` into `n_splits` different parts,
    and selects a different part as validation and testing set.  The training
    set is made up of the remaining `n_splits`-2 parts.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True)
    # We have to explicitly force `dtype=np.object_` so this doesn't fail for
    # folds of different sizes
    folds = np.array([fold for _, fold in splitter.split(items)], dtype=np.object_)
    for test_fold, test_fold_idxs in enumerate(folds):
        # We have to agressively do `astype()`s here, as, if all folds have the
        # same size, the folds get coerced into one 2D tensor with dtype
        # `object` instead of one with dtype int
        test_fold_idxs = test_fold_idxs.astype(int)
        val_fold = (test_fold + 1) % n_splits
        val_fold_idxs = folds[val_fold].astype(int)

        train_folds = set(range(n_splits)) - {test_fold, val_fold}
        train_fold_idxs = np.concatenate(folds[list(train_folds)]).astype(int)

        yield (
            items[train_fold_idxs],
            items[val_fold_idxs],
            items[test_fold_idxs],
        )


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def app(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    dataset_df = load_dataset_df(cfg)
    crossval_id = None

    for fold_no, (train_idx, valid_idx, test_idx) in enumerate(
        get_splits(dataset_df.index.values, n_splits=5)
    ):
        print("=" * 50)
        print(f"Fold {fold_no}")
        print("=" * 50)

        train_df, valid_df, test_df = (
            dataset_df.loc[train_idx],
            dataset_df.loc[valid_idx],
            dataset_df.loc[test_idx],
        )

        print("Train dataset:")
        print(indent(summarize_dataset(cfg.dataset.targets, train_df), "  "))

        assert not (
            overlap := set(train_df.index) & set(valid_df.index)
        ), f"unexpected overlap between training and testing set: {overlap}"

        encoders = {
            target.column: TargetEncoder.for_target(target)
            for target in cfg.dataset.targets
        }
        train_targets = {t: encoder.fit(train_df) for t, encoder in encoders.items()}
        valid_targets = {t: encoder(valid_df) for t, encoder in encoders.items()}
        test_targets = {t: encoder(test_df) for t, encoder in encoders.items()}

        train_ds = BagDataset(
            bags=train_df.path.values,
            targets=train_targets,
            instances_per_bag=cfg.dataset.instances_per_bag,
            pad=False,
            deterministic=False,
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        valid_ds = BagDataset(
            bags=valid_df.path.values,
            targets=valid_targets,
            instances_per_bag=cfg.dataset.instances_per_bag,
            pad=False,
            deterministic=True,
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
        )
        test_ds = BagDataset(
            bags=test_df.path.values,
            targets=test_targets,
            instances_per_bag=cfg.dataset.instances_per_bag,
            pad=False,
            deterministic=True,
        )
        test_dl = DataLoader(
            test_ds,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
        )

        model, trainer, out_dir, wandb_logger = train(
            cfg,
            train_dl,
            valid_dl,
            crossval_id=crossval_id or "",
            crossval_fold=fold_no,
        )

        if crossval_id is None:
            crossval_id = wandb_logger.experiment.config["crossval_id"]

        trainer.test(model=model, dataloaders=test_dl)

        for dl, df, filename in [
            (valid_dl, valid_df, "valid-patient-preds.csv"),
            (test_dl, test_df, "patient-preds.csv"),
        ]:
            predictions = flatten_batched_dicts(
                trainer.predict(model=model, dataloaders=dl)
            )
            preds_df = make_preds_df(
                predictions=predictions,
                base_df=df,
                categories={
                    target.column: target.classes for target in cfg.dataset.targets
                },
            )
            preds_df.to_csv(out_dir / filename)

        wandb_logger.experiment.finish()


if __name__ == "__main__":
    app()
