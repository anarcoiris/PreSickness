"""
Archivo wrapper para retro-compatibilidad.
La implementación real vive en `services/api-gateway/main.py`.
"""
from importlib import util
from pathlib import Path

_SERVICE_PATH = Path(__file__).parent / "services" / "api-gateway" / "main.py"
_SPEC = util.spec_from_file_location("api_gateway_service", _SERVICE_PATH)
_MODULE = util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)  # type: ignore[arg-type]
app = getattr(_MODULE, "app")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_gateway:app", host="0.0.0.0", port=8000, reload=True)
