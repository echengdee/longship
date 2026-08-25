from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np


def resolve_onnx_providers(requested: str = "auto") -> tuple[str, ...]:
    """Resolve one platform-wide ONNX Runtime provider policy."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    if requested == "cpu":
        providers = ("CPUExecutionProvider",)
    elif requested == "cuda":
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
    elif requested == "auto":
        providers = (
            ("CUDAExecutionProvider", "CPUExecutionProvider")
            if "CUDAExecutionProvider" in available
            else ("CPUExecutionProvider",)
        )
    else:
        raise ValueError(f"unknown ONNX Runtime provider policy {requested!r}")
    missing = set(providers) - available
    if requested == "cuda" and "CUDAExecutionProvider" in missing:
        raise RuntimeError("CUDAExecutionProvider was requested but is unavailable")
    return tuple(provider for provider in providers if provider in available)


class OnnxEngine:
    """Small validated wrapper used by every Longship ONNX policy pipeline."""

    def __init__(
        self,
        model: str | Path,
        *,
        provider: str = "auto",
        intra_op_threads: int = 0,
    ) -> None:
        import onnxruntime as ort

        self.model = Path(model)
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            options.intra_op_num_threads = intra_op_threads
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(self.model),
            sess_options=options,
            providers=list(resolve_onnx_providers(provider)),
        )
        self.load_seconds = time.perf_counter() - started
        self.input_names = tuple(value.name for value in self.session.get_inputs())
        self.output_names = tuple(value.name for value in self.session.get_outputs())
        self.last_inference_seconds = 0.0

    @property
    def providers(self) -> tuple[str, ...]:
        return tuple(self.session.get_providers())

    def infer(
        self,
        inputs: Mapping[str, np.ndarray],
        output_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, ...]:
        missing = set(self.input_names) - set(inputs)
        unknown = set(inputs) - set(self.input_names)
        if missing or unknown:
            raise ValueError(
                f"ONNX input mismatch for {self.model.name}: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        started = time.perf_counter()
        outputs = self.session.run(None if output_names is None else list(output_names), dict(inputs))
        self.last_inference_seconds = time.perf_counter() - started
        result = tuple(np.asarray(value) for value in outputs)
        if any(not np.all(np.isfinite(value)) for value in result if value.dtype.kind == "f"):
            raise FloatingPointError(f"{self.model.name} produced NaN or infinity")
        return result
