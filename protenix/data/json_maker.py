"""Backwards-compatible imports for older downstream code (e.g. ProtenixScore)."""

from protenix.data.inference.json_maker import atom_array_to_input_json, cif_to_input_json

__all__ = ["atom_array_to_input_json", "cif_to_input_json"]

