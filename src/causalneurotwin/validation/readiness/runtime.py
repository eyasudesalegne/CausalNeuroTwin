"""Optional TVB bootstrap. Never imported during ordinary synthetic unit tests."""

import importlib.metadata
import os


def tvb_runtime(c, out):
    # This adapter has not been run against participant data or the remote TVB runtime.
    if importlib.metadata.version("tvb-library") != c["wp3"]["tvb_version"]:
        raise RuntimeError("TVB version drift")
    os.environ["NUMBA_CACHE_DIR"] = str(out / ".numba")
    os.environ["TVB_USER_HOME"] = str(out / ".tvb")
    (out / ".tvb").mkdir(exist_ok=False)
    # TVB may replace console configuration; its library.log is under TVB_USER_HOME.
    import logging

    logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler()], force=True)
    from tvb.basic.profile import TvbProfile

    TvbProfile.set_profile(TvbProfile.LIBRARY_PROFILE, run_init=True)
    from tvb.datatypes.connectivity import Connectivity
    from tvb.simulator import coupling, integrators, models, monitors, simulator

    return Connectivity, simulator, models, coupling, integrators, monitors
