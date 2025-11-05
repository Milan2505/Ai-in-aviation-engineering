"""Utilities for the Ai-in-aviation-engineering project."""

from .io_utils import dedup_names, is_potential_multi_index
from .ml_pipeline import (
    DATA_DIR,
    MODELS_DIR,
    MaintenanceModelArtifacts,
    PriceModelArtifacts,
    load_data,
    load_maintenance_artifacts,
    load_price_artifacts,
    predict_maintenance,
    predict_price,
    train_maintenance_model,
    train_price_model,
)

__all__ = [
    "dedup_names",
    "is_potential_multi_index",
    "DATA_DIR",
    "MODELS_DIR",
    "PriceModelArtifacts",
    "MaintenanceModelArtifacts",
    "load_data",
    "train_price_model",
    "train_maintenance_model",
    "predict_price",
    "predict_maintenance",
    "load_price_artifacts",
    "load_maintenance_artifacts",
]
