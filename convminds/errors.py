from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from convminds.data.types import DataCategory


class ConvMindsError(Exception):
    """Base error for all convminds logic and validation failures."""
    pass


class DataTraitMismatchError(ConvMindsError, TypeError):
    """
    Raised when an operation (e.g., a transform) is incompatible with the DataCategory.
    Example: Applying an HRF sliding window to stimulus-averaged beta maps.
    """
    def __init__(self, operation_name: str, required: DataCategory, found: DataCategory | None):
        found_name = found.name if found else "UNKNOWN"
        super().__init__(
            f"Operation '{operation_name}' requires trait {required.name}, "
            f"but received data categorized as {found_name}. "
            f"This is often a logic error where temporal windowing is applied to stimulus-averaged (beta-map) data."
        )


class SpatialGroundingError(ConvMindsError):
    """
    Raised when a spatial operation (e.g., SpatialAttentionEncoder) is attempted 
    on data that is missing coordinates (MNI/ROI).
    """
    pass


class ResourceNotReadyError(ConvMindsError, FileNotFoundError):
    """
    Raised when processed data or model activations are accessed before being properly initialized or cached.
    """
    pass
