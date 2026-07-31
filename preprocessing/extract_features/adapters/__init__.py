from __future__ import annotations

from .bendr import BENDRAdapter
from .biot import BIOTAdapter
from .cbramod import CBraModAdapter
from .labram import LaBraMAdapter
from .singlem import SingLEMAdapter


def _direct(cls):
    return lambda context, device: cls(context, device)


def _benchmark(model_name: str):
    from .benchmark import BenchmarkAdapter

    return lambda context, device: BenchmarkAdapter(model_name, context, device)


ADAPTERS = {
    "singlem": _direct(SingLEMAdapter),
    "bendr": _direct(BENDRAdapter),
    "biot": _direct(BIOTAdapter),
    "cbramod": _direct(CBraModAdapter),
    "labram": _direct(LaBraMAdapter),
    "csbrain": _benchmark("csbrain"),
    "codebrain": _benchmark("codebrain"),
    "luna_large": _benchmark("luna_large"),
    "mirepnet": _benchmark("mirepnet"),
}


def model_names() -> list[str]:
    return sorted(ADAPTERS)


def build_adapter(name: str, context: dict, device):
    return ADAPTERS[name](context, device)
