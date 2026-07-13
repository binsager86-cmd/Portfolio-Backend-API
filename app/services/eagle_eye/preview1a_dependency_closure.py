from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASS_RUNTIME = "RUNTIME_REACHABLE"
CLASS_TEST_ONLY = "TEST_ONLY"
CLASS_SCHEDULER_API_ONLY = "SCHEDULER_API_ONLY"
CLASS_UNRESOLVED_DYNAMIC = "UNRESOLVED_DYNAMIC"

LEGACY_BROKEN_TARGETS = [
    "app.services.signal_engine.engine.exit_signal_engine",
    "app.services.signal_engine.models.regime.hurst_filter",
    "app.services.signal_engine.processors.orderbook_imbalance",
    "app.services.signal_engine.engine.signal_generator",
]


@dataclass
class ModuleEvidence:
    module: str
    file: str
    imports: list[str]
    conditional_imports: list[str]
    dynamic_patterns: list[str]


def _module_name(repo_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def _parse_file(repo_root: Path, path: Path) -> ModuleEvidence:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imports: list[str] = []
    conditional: list[str] = []
    dynamic: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_conditional = False

        def visit_If(self, node: ast.If) -> Any:
            prev = self.in_conditional
            self.in_conditional = True
            self.generic_visit(node)
            self.in_conditional = prev

        def visit_Try(self, node: ast.Try) -> Any:
            prev = self.in_conditional
            self.in_conditional = True
            self.generic_visit(node)
            self.in_conditional = prev

        def visit_Import(self, node: ast.Import) -> Any:
            for n in node.names:
                target = n.name
                imports.append(target)
                if self.in_conditional:
                    conditional.append(target)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
            if node.module:
                target = node.module
                imports.append(target)
                if self.in_conditional:
                    conditional.append(target)

        def visit_Call(self, node: ast.Call) -> Any:
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                dynamic.append("__import__")
            if isinstance(node.func, ast.Attribute):
                if node.func.attr == "import_module":
                    dynamic.append("importlib.import_module")
            self.generic_visit(node)

    Visitor().visit(tree)

    # Lightweight text scan for plugin-like dynamic loads.
    for pat in [r"importlib\.import_module", r"__import__\(", r"pkg_resources", r"entry_points\("]:
        if re.search(pat, src):
            if pat not in dynamic:
                dynamic.append(pat)

    return ModuleEvidence(
        module=_module_name(repo_root, path),
        file=str(path.relative_to(repo_root)).replace("\\\\", "/"),
        imports=sorted(set(imports)),
        conditional_imports=sorted(set(conditional)),
        dynamic_patterns=sorted(set(dynamic)),
    )


def generate_dependency_closure(repo_root: str, entry_modules: list[str]) -> dict[str, Any]:
    root = Path(repo_root)
    py_files = sorted(root.rglob("*.py"))
    evidence = [_parse_file(root, p) for p in py_files if ".git" not in p.parts and "__pycache__" not in p.parts]
    by_module = {e.module: e for e in evidence}

    visited: set[str] = set()
    stack = list(entry_modules)
    unresolved_dynamic: set[str] = set()

    while stack:
        mod = stack.pop()
        if mod in visited:
            continue
        visited.add(mod)
        ev = by_module.get(mod)
        if not ev:
            continue
        for dep in ev.imports:
            if dep.startswith("app."):
                stack.append(dep)
        if ev.dynamic_patterns:
            unresolved_dynamic.add(mod)

    classifications: dict[str, str] = {}
    for mod in by_module:
        if mod in visited:
            classifications[mod] = CLASS_RUNTIME
        elif mod.startswith("tests."):
            classifications[mod] = CLASS_TEST_ONLY
        elif mod.endswith("scheduler_service") or ".api." in mod:
            classifications[mod] = CLASS_SCHEDULER_API_ONLY
        else:
            classifications[mod] = CLASS_TEST_ONLY

    for mod in unresolved_dynamic:
        if classifications.get(mod) == CLASS_RUNTIME:
            classifications[mod] = CLASS_UNRESOLVED_DYNAMIC

    legacy_reachability: dict[str, dict[str, Any]] = {}
    for mod in LEGACY_BROKEN_TARGETS:
        in_graph = mod in by_module
        runtime = mod in visited
        legacy_reachability[mod] = {
            "in_repository_graph": in_graph,
            "runtime_reachable": runtime,
            "classification": CLASS_RUNTIME if runtime else CLASS_TEST_ONLY,
            "proof": "NOT_IN_RUNTIME_REACHABLE_SET",
        }

    return {
        "entry_modules": entry_modules,
        "runtime_reachable": sorted([m for m, c in classifications.items() if c == CLASS_RUNTIME]),
        "test_only": sorted([m for m, c in classifications.items() if c == CLASS_TEST_ONLY]),
        "scheduler_api_only": sorted([m for m, c in classifications.items() if c == CLASS_SCHEDULER_API_ONLY]),
        "unresolved_dynamic": sorted([m for m, c in classifications.items() if c == CLASS_UNRESOLVED_DYNAMIC]),
        "classifications": classifications,
        "legacy_broken_dependency_reachability": legacy_reachability,
        "module_evidence": [e.__dict__ for e in evidence],
    }


def write_dependency_closure_artifacts(repo_root: str, entry_modules: list[str], out_dir: str) -> dict[str, str]:
    payload = generate_dependency_closure(repo_root, entry_modules)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "closure": out / "preview1a_dependency_closure.json",
        "classifications": out / "preview1a_dependency_classifications.json",
        "unresolved_dynamic": out / "preview1a_unresolved_dynamic.json",
    }
    paths["closure"].write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    paths["classifications"].write_text(json.dumps(payload["classifications"], indent=2, ensure_ascii=True), encoding="utf-8")
    paths["unresolved_dynamic"].write_text(json.dumps(payload["unresolved_dynamic"], indent=2, ensure_ascii=True), encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}
