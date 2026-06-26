"""Quiet Hugging Face / Transformers startup noise for local embedding models."""
from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout


def configure_quiet_hf() -> None:
    """Suppress non-actionable model-loading warnings and progress bars."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    for name in (
        "huggingface_hub",
        "huggingface_hub.file_download",
        "sentence_transformers",
        "transformers",
        "safetensors",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
    warnings.filterwarnings("ignore", message=".*position_ids.*")

    try:
        from huggingface_hub.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bars()
    except Exception:
        pass

    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except Exception:
        pass


@contextmanager
def quiet_model_load():
    """Redirect noisy model-loader progress output while preserving exceptions."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield
