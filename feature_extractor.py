"""
Archivo wrapper para mantener compatibilidad con tooling existente.
La implementación oficial está en `services/feature-extractor/worker.py`.
"""
from importlib import util
from pathlib import Path

_SERVICE_PATH = Path(__file__).parent / "services" / "feature-extractor" / "worker.py"
_SPEC = util.spec_from_file_location("feature_extractor_worker", _SERVICE_PATH)
_MODULE = util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]


if __name__ == "__main__":
    from asyncio import run

    run(_MODULE.main())  # type: ignore[attr-defined]
