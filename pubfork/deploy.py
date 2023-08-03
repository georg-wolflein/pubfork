from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra
import os
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"

from .utils import (
    flatten_batched_dicts,
    make_preds_df,
)
from .data import BagDataset
from .targets import TargetEncoder
from .train import load_dataset_df, LitMilTransformer


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def app(cfg: DictConfig) -> None:
    out_dir = Path(cfg.output_dir)
    dataset_df = load_dataset_df(cfg)

    # Make a dataset with faux labels (the labels will be ignored)
    ds = BagDataset(
        bags=dataset_df.path.values,
        targets={},
        instances_per_bag=None,
        pad=False,
        deterministic=True,
    )
    dl = DataLoader(
        ds,
        shuffle=False,
        batch_size=1,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
    )

    model = LitMilTransformer.load_from_checkpoint(cfg.deploy.checkpoint)

    trainer = pl.Trainer(
        default_root_dir=out_dir,
        accelerator="gpu",
        devices=cfg.device or 1,
        logger=False,
    )
    predictions = flatten_batched_dicts(trainer.predict(model=model, dataloaders=dl))
    preds_df = make_preds_df(
        predictions=predictions,
        base_df=dataset_df.drop(columns="path"),
        categories={target.column: target.classes for target in model.hparams.targets},
    )
    out_file = out_dir / "patient-preds.csv"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    preds_df.to_csv(out_dir / "patient-preds.csv")

    print(f"Saved predictions to {out_file}")


if __name__ == "__main__":
    app()
