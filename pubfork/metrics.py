from typing import Type
from torchmetrics import Metric, MetricCollection
from torch import Tensor
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    MulticlassAUROC,
    MulticlassAveragePrecision,
)

__all__ = [
    "ClasswiseMulticlassAUROC",
    "ClasswiseMulticlassAveragePrecision",
    "create_metrics_for_target",
]


def _make_classwise_metric(metric_class: Type[Metric]) -> Type[Metric]:
    class ClasswiseMetric(metric_class):
        def __init__(self, *args, class_id, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_id = class_id

        def update(self, preds: Tensor, target: Tensor):
            super().update(preds[..., self.class_id], target == self.class_id)

    return ClasswiseMetric


ClasswiseMulticlassAUROC = _make_classwise_metric(BinaryAUROC)
ClasswiseMulticlassAveragePrecision = _make_classwise_metric(BinaryAveragePrecision)


def create_metrics_for_target(target) -> MetricCollection:
    if target.type == "categorical":
        multiclass_metrics = {
            "auroc": MulticlassAUROC,
            "ap": MulticlassAveragePrecision,
        }
        classwise_metrics = {
            "auroc": ClasswiseMulticlassAUROC,
            "ap": ClasswiseMulticlassAveragePrecision,
        }
        return MetricCollection(
            {
                **{
                    name: metric(num_classes=len(target.classes))
                    for name, metric in multiclass_metrics.items()
                },
                **{
                    f"{name}_{c}": metric(class_id=i)
                    for i, c in enumerate(target.classes)
                    for name, metric in classwise_metrics.items()
                },
            }
        )
    else:
        raise NotImplementedError(f"Unknown target type {target.type}")
