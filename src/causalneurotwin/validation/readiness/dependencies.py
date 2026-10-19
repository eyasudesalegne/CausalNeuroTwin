"""Mode-specific mandatory dependencies; optional absence is explicit metadata."""

import importlib.metadata as metadata


def dependency_report(mode, version=None):
    if mode not in ("audit", "intrinsic", "smoke"):
        raise ValueError("Unknown dependency mode")
    version = metadata.version if version is None else version
    required = ["numpy", "scipy"] + (["nibabel"] if mode == "audit" else ["tvb-library", "numba"])
    optional = [] if mode == "audit" else ["nibabel"]
    report = {}
    for name in required + optional:
        try:
            report[name] = {
                "status": "present",
                "version": version(name),
                "required": name in required,
            }
        except metadata.PackageNotFoundError as error:
            if name in required:
                raise RuntimeError("Required dependency missing: " + name) from error
            report[name] = {"status": "not_installed", "version": None, "required": False}
    return report
