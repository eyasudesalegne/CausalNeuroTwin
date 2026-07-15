"""Environment diagnostics for the CausalNeuroTwin repository foundation."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from causalneurotwin import __version__

_OPTIONAL_MODULES: tuple[str, ...] = (
    "numpy",
    "torch",
    "nibabel",
    "mpi4py",
    "simnibs",
    "tvb",
)


@dataclass(frozen=True)
class DataRootStatus:
    """Privacy-preserving status of the optional local data root."""

    configured: bool
    status: str
    readable: bool


@dataclass(frozen=True)
class OutputStatus:
    """Writable-output check without exposing the absolute path."""

    writable: bool
    status: str


@dataclass(frozen=True)
class AcceleratorStatus:
    """Optional PyTorch/CUDA status."""

    torch_available: bool
    cuda_available: bool
    cuda_device_count: int
    device_names: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class DoctorReport:
    """Machine-readable repository-foundation diagnostic."""

    package: str
    package_version: str
    python_version: str
    python_supported: bool
    operating_system: str
    machine: str
    optional_dependencies: dict[str, bool]
    data_root: DataRootStatus
    output: OutputStatus
    accelerator: AcceleratorStatus
    overall_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return asdict(self)


def _module_available(name: str) -> bool:
    """Return whether an importable module specification exists."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def inspect_optional_dependencies() -> dict[str, bool]:
    """Inspect optional scientific dependencies without importing them."""

    return {name: _module_available(name) for name in _OPTIONAL_MODULES}


def inspect_data_root(data_root: Path | None) -> DataRootStatus:
    """Inspect a data root while returning no path information."""

    if data_root is None:
        return DataRootStatus(configured=False, status="not_configured", readable=False)
    if not data_root.exists():
        return DataRootStatus(configured=True, status="missing", readable=False)
    if not data_root.is_dir():
        return DataRootStatus(configured=True, status="not_a_directory", readable=False)
    readable = os.access(data_root, os.R_OK | os.X_OK)
    status = "ready" if readable else "not_readable"
    return DataRootStatus(configured=True, status=status, readable=readable)


def inspect_output(output_dir: Path) -> OutputStatus:
    """Test directory creation and file writing, then remove the probe file."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="cnt-doctor-",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as handle:
            handle.write("CausalNeuroTwin writable-output probe.\n")
            probe = Path(handle.name)
        probe.unlink(missing_ok=True)
        return OutputStatus(writable=True, status="ready")
    except OSError:
        return OutputStatus(writable=False, status="not_writable")


def inspect_accelerator(torch_available: bool) -> AcceleratorStatus:
    """Inspect CUDA safely when PyTorch is available."""

    if not torch_available:
        return AcceleratorStatus(
            torch_available=False,
            cuda_available=False,
            cuda_device_count=0,
            device_names=(),
            status="torch_not_installed",
        )

    try:
        torch = importlib.import_module("torch")

        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        names = tuple(str(torch.cuda.get_device_name(index)) for index in range(count))
        status = "cuda_ready" if cuda_available else "cuda_unavailable"
        return AcceleratorStatus(
            torch_available=True,
            cuda_available=cuda_available,
            cuda_device_count=count,
            device_names=names,
            status=status,
        )
    except Exception as exc:  # pragma: no cover - depends on optional binary runtime
        return AcceleratorStatus(
            torch_available=True,
            cuda_available=False,
            cuda_device_count=0,
            device_names=(),
            status=f"torch_runtime_error:{type(exc).__name__}",
        )


def resolve_data_root(explicit: Path | None = None) -> Path | None:
    """Resolve the optional data root from an argument or environment variable."""

    if explicit is not None:
        return explicit
    raw = os.environ.get("CAUSALNEUROTWIN_DATA_ROOT")
    return Path(raw).expanduser() if raw else None


def resolve_output_dir(explicit: Path | None = None) -> Path:
    """Resolve an output directory without requiring configuration."""

    if explicit is not None:
        return explicit
    raw = os.environ.get("CAUSALNEUROTWIN_OUTPUT_ROOT")
    return Path(raw).expanduser() if raw else Path(".causalneurotwin-doctor")


def build_report(data_root: Path | None, output_dir: Path) -> DoctorReport:
    """Build the complete diagnostic report."""

    optional = inspect_optional_dependencies()
    output = inspect_output(output_dir)
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    accelerator = inspect_accelerator(optional["torch"])
    overall = "ready" if output.writable and python_supported else "attention_required"
    return DoctorReport(
        package="causalneurotwin",
        package_version=__version__,
        python_version=platform.python_version(),
        python_supported=python_supported,
        operating_system=platform.system(),
        machine=platform.machine(),
        optional_dependencies=optional,
        data_root=inspect_data_root(data_root),
        output=output,
        accelerator=accelerator,
        overall_status=overall,
    )


def render_human(report: DoctorReport) -> str:
    """Render a concise human-readable diagnostic."""

    dependencies = ", ".join(
        f"{name}={'yes' if available else 'no'}"
        for name, available in report.optional_dependencies.items()
    )
    lines = [
        f"CausalNeuroTwin {report.package_version}",
        (
            f"Python: {report.python_version} "
            f"({'supported' if report.python_supported else 'unsupported'})"
        ),
        f"Platform: {report.operating_system} / {report.machine}",
        f"Optional dependencies: {dependencies}",
        f"Data root: {report.data_root.status}",
        f"Output: {report.output.status}",
        f"Accelerator: {report.accelerator.status}",
        f"Overall: {report.overall_status}",
    ]
    return "\n".join(lines)
