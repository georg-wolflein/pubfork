from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple
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
from hydra.core.global_hydra import GlobalHydra
import os
from textwrap import indent

os.environ["HYDRA_FULL_ERROR"] = "1"

from pubfork.utils import (
    make_dataset_df,
    pathlist,
    DummyBiggestBatchFirstCallback,
    flatten_batched_dicts,
    make_preds_df,
)
from pubfork.data import BagDataset
from pubfork.model import LitMilClassificationMixin
from pubfork.targets import TargetEncoder

# GlobalHydra().clear()
# hydra.initialize(config_path="conf")
# cfg = hydra.compose(
#     "config.yaml"#, overrides=["+experiment=mnist_collage", "+model=gnn_gat"]
# )


class LitMilTransformer(LitMilClassificationMixin):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(
            targets=cfg.dataset.targets,
            learning_rate=cfg.learning_rate,
        )
        self.model = hydra.utils.instantiate(cfg.model)

        self.save_hyperparameters()

    def forward(self, *args):
        return self.model(*args)


class DefineWandbMetricsCallback(Callback):
    def __init__(self, model: LitMilClassificationMixin, run) -> None:
        super().__init__()
        self.metric_goals = model.metric_goals
        self.run = run

    def on_train_start(self, *args, **kwargs) -> None:
        for name, goal in self.metric_goals.items():
            self.run.define_metric(name, goal=f"{goal}imize")


def summarize_dataset(targets: ListConfig, df: pd.DataFrame):
    ret = f"Number of patients: {len(df)}\n"
    for target in targets:
        target_info = f"Target: {target.column} ({target.type})\n"
        if target.type == "categorical":
            value_counts = df[target.column].value_counts(normalize=True)
            inv_value_counts = 1.0 / value_counts
            inv_value_counts /= inv_value_counts.sum()
            weights = target.weights or ([1.0] * len(target.classes))
            weights = pd.Series(weights, index=target.classes)
            merged = (
                value_counts.to_frame("freq")
                .join(inv_value_counts.to_frame("inv freq"), how="outer")
                .join(weights.to_frame("weight"), how="outer")
            )
            target_info += f"  {merged.to_string(float_format=lambda x: f'{x:.4f}')}\n"
        elif target.type == "continuous":
            target_info += df[target.column].describe().to_string()
        ret += indent(target_info, "  ")
    return ret


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def app(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    torch.set_float32_matmul_precision("medium")

    dataset_df = make_dataset_df(
        clini_tables=pathlist(cfg.dataset.clini_tables),
        slide_tables=pathlist(cfg.dataset.slide_tables),
        feature_dirs=pathlist(cfg.dataset.feature_dirs),
        patient_col=cfg.dataset.patient_col,
        filename_col=cfg.dataset.filename_col,
        target_labels=[label.column for label in cfg.dataset.targets],
    )

    # Remove patients with no target labels
    to_delete = pd.Series(False, index=dataset_df.index)
    for target in cfg.dataset.targets:
        to_delete |= dataset_df[target.column].isna()
    if to_delete.any():
        print(
            f"Removing {to_delete.sum()} patients with missing target labels; {(~to_delete).sum()} remaining"
        )
    dataset_df = dataset_df[~to_delete]

    # Split validation set off main dataset
    train_items, valid_items = train_test_split(dataset_df.index, test_size=0.2)
    train_df, valid_df = dataset_df.loc[train_items], dataset_df.loc[valid_items]

    print("Train dataset:")
    print(indent(summarize_dataset(cfg.dataset.targets, train_df), "  "))

    print("Validation dataset:")
    print(indent(summarize_dataset(cfg.dataset.targets, valid_df), "  "))

    assert not (
        overlap := set(train_df.index) & set(valid_df.index)
    ), f"unexpected overlap between training and testing set: {overlap}"

    encoders = {
        target.column: TargetEncoder.for_target(target)
        for target in cfg.dataset.targets
    }
    train_targets = {t: encoder.fit(train_df) for t, encoder in encoders.items()}
    valid_targets = {t: encoder(valid_df) for t, encoder in encoders.items()}

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

    model = LitMilTransformer(cfg)

    wandb_logger = WandbLogger(cfg.name, project=cfg.project)
    wandb_logger.log_hyperparams(
        {
            **OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            "overrides": " ".join(sys.argv[1:]),
        }
    )
    wandb_logger.experiment.log_code(
        ".",
        include_fn=lambda path: Path(path).suffix in {".py", ".yaml", ".yml"}
        and "env" not in Path(path).parts,
    )
    out_dir = Path(cfg.output_dir) / (wandb_logger.version or "")

    define_metrics_callback = DefineWandbMetricsCallback(model, wandb_logger.experiment)

    trainer = pl.Trainer(
        # profiler="simple",
        default_root_dir=out_dir,
        callbacks=[
            EarlyStopping(monitor="val/loss", mode="min", patience=cfg.patience),
            ModelCheckpoint(
                monitor="val/loss",
                mode="min",
                filename="checkpoint-{epoch:02d}-{val/loss:0.3f}",
            ),
            DummyBiggestBatchFirstCallback(
                train_dl.dataset.dummy_batch(cfg.dataset.batch_size)
            ),
            define_metrics_callback,
        ],
        max_epochs=cfg.max_epochs,
        # FIXME The number of accelerators is currently fixed to one for the
        # following reasons:
        #  1. `trainer.predict()` does not return any predictions if used with
        #     the default strategy no multiple GPUs
        #  2. `barspoon.model.SafeMulticlassAUROC` breaks on multiple GPUs.
        accelerator="gpu",
        devices=cfg.device or 1,
        accumulate_grad_batches=cfg.accumulate_grad_samples // cfg.dataset.batch_size,
        gradient_clip_val=cfg.grad_clip,
        logger=[CSVLogger(save_dir=out_dir), wandb_logger],
    )

    print(model)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    predictions = flatten_batched_dicts(
        trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True)
    )

    preds_df = make_preds_df(
        predictions=predictions,
        base_df=valid_df,
        categories={target.column: target.classes for target in cfg.dataset.targets},
    )
    preds_df.to_csv(out_dir / "valid-patient-preds.csv")


if __name__ == "__main__":
    app()
