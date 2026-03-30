from __future__ import annotations

from importlib import import_module


def _require_nibabel():
    try:
        import nibabel as nib
    except ModuleNotFoundError as error:
        raise RuntimeError("nibabel is required for NIfTI loading. Install it via pip.") from error
    return nib


def _require_scipy():
    try:
        import scipy.io as sio
    except ModuleNotFoundError as error:
        raise RuntimeError("scipy is required for MATLAB (.mat) loading. Install it via pip.") from error
    return sio


def _require_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("pandas is required for TSV parsing. Install it via pip.") from error
    return pd


def _resolve_transformers():
    try:
        return import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError("HFArtificialSubject requires transformers to be installed.") from error
