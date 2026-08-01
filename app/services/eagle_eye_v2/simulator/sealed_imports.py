from __future__ import annotations

import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from app.services.eagle_eye_v2.simulator.constants import FROZEN_CODE

BACKEND_API_ROOT = Path(__file__).resolve().parents[4]
RELEASE_ROOT = BACKEND_API_ROOT.parent / "backend-api-main-release"
RELEASE_SCRIPTS = RELEASE_ROOT / "scripts"
RELEASE_EE_V2_PACKAGE = RELEASE_ROOT / "app" / "services" / "eagle_eye_v2"


class FrozenImportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_file(name: str) -> Path:
    spec = FROZEN_CODE[name]
    path = Path(spec["path"])
    if not path.exists():
        raise FrozenImportError(f"frozen {name} file missing: {path}")
    actual = sha256_file(path)
    expected = str(spec["sha256"])
    if actual.lower() != expected.lower():
        raise FrozenImportError(f"frozen {name} SHA mismatch: {actual} != {expected}")
    return path


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise FrozenImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_harness_dependency_paths() -> None:
    for path in (assert_frozen_file("harness").parent, RELEASE_SCRIPTS, RELEASE_ROOT):
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
    if RELEASE_EE_V2_PACKAGE.exists():
        import app.services.eagle_eye_v2 as ee_v2_package

        package_path = str(RELEASE_EE_V2_PACKAGE)
        if package_path not in ee_v2_package.__path__:
            ee_v2_package.__path__.append(package_path)


@lru_cache(maxsize=1)
def load_state_machine() -> ModuleType:
    return _load_module("eagle_eye_freeze_v3_r16_3_candidate_state_machine", assert_frozen_file("state_machine"))


@lru_cache(maxsize=1)
def load_harness_layer1() -> ModuleType:
    _prepare_harness_dependency_paths()
    sys.modules["r16_3_candidate_state_machine"] = load_state_machine()
    return _load_module("eagle_eye_freeze_v3_r16_3_harness_v53", assert_frozen_file("harness"))


def verify_frozen_imports() -> dict[str, str]:
    return {name: sha256_file(assert_frozen_file(name)) for name in FROZEN_CODE}
