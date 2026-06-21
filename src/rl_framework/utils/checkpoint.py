"""Checkpoint path resolution and validation utilities.

All model and VecNormalize path logic lives here so training, evaluation, and
reproducibility code share one canonical resolution chain rather than each
re-implementing it.
"""

from __future__ import annotations

from pathlib import Path


def model_zip_path(path: str | Path) -> Path:
    """Return *path* as a ``Path``, appending ``.zip`` if not already present."""
    path = Path(path)
    return path if str(path).endswith(".zip") else Path(str(path) + ".zip")


def vecnormalize_path_for_model(model_path: str | Path) -> Path:
    """Return the model-specific VecNormalize sidecar path.

    Periodic checkpoints need their own normaliser snapshot because a shared
    ``vecnormalize.pkl`` can drift after the model was written.
    """
    path = model_zip_path(model_path)
    return path.with_name(path.stem + "_vecnormalize.pkl")


def legacy_vecnormalize_path_for_model(model_path: str | Path) -> Path:
    """Return the legacy shared ``vecnormalize.pkl`` path next to *model_path*."""
    return model_zip_path(model_path).with_name("vecnormalize.pkl")


def find_vecnormalize_path_for_model(model_path: str | Path) -> Path | None:
    """Discover the VecNormalize file for *model_path* using a three-tier chain.

    Search order:
    1. Model-specific sidecar (``<stem>_vecnormalize.pkl``)
    2. Legacy shared sidecar (``vecnormalize.pkl`` next to the model)
    3. ``None`` — no normaliser found
    """
    specific = vecnormalize_path_for_model(model_path)
    if specific.exists():
        return specific
    legacy = legacy_vecnormalize_path_for_model(model_path)
    if legacy.exists():
        return legacy
    return None


def validate_resume_path(resume_from: Path, normalize: bool) -> None:
    """Raise early with a clear message if resume files are missing or corrupt."""
    import zipfile

    model_path = model_zip_path(resume_from)
    if not model_path.exists():
        raise FileNotFoundError(
            f"resume_from model not found: {model_path}. "
            "Check that the checkpoint path is correct."
        )
    if not zipfile.is_zipfile(model_path):
        raise ValueError(
            f"Checkpoint {model_path} appears corrupt (not a valid zip file). "
            "The file may have been truncated by an interrupted write."
        )
    if normalize:
        if find_vecnormalize_path_for_model(model_path) is None:
            raise FileNotFoundError(
                f"VecNormalize sidecar not found for model {model_path}. "
                f"Expected {vecnormalize_path_for_model(model_path).name} "
                f"(or legacy {legacy_vecnormalize_path_for_model(model_path).name}). "
                "Move the model and normalizer together, or set "
                "normalize_observations: false."
            )
