"""Backwards-compatible imports for older downstream code (e.g. ProtenixScore).

Upstream Protenix reorganized data modules under `protenix.data.core` and
`protenix.data.inference`. Keep thin re-exports here so forks/downstream tools
can continue importing from the old paths.
"""

from protenix.data.core.filter import Filter

__all__ = ["Filter"]

