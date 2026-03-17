"""Backwards-compatible imports for older downstream code (e.g. ProtenixScore)."""

from protenix.data.inference.infer_dataloader import InferenceDataset, get_inference_dataloader

__all__ = ["InferenceDataset", "get_inference_dataloader"]

