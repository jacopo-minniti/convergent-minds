from __future__ import annotations
from enum import Enum, auto


class DataCategory(Enum):
    """
    Distinguish between different data structures to prevent incompatible operations.
    
    STIMULUS_LEVEL: One brain vector per sentence/concept (e.g. Pereira).
                    Signal shape is (N_samples, 1, N_voxels).
    
    TOKEN_LEVEL: Time-series fMRI data aligned for word-by-word progression.
                 Signal shape is (N_samples, T, N_voxels).
    """
    STIMULUS_LEVEL = auto()
    TOKEN_LEVEL = auto()


def check_trait(brain_data, required_category: DataCategory, operation_name: str = "Unspecified"):
    """Raise a DataTraitMismatchError if the brain data does not match the required category."""
    from convminds.errors import DataTraitMismatchError
    
    found = getattr(brain_data, "category", None)
    if found is not None and found != required_category:
        raise DataTraitMismatchError(operation_name, required_category, found)
