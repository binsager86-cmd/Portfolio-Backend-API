from app.api.v1 import v1_router
from app.main import app


def test_app_routes_expose_path_for_included_routers():
    route_types_missing_path_attr = [
        type(route).__name__ for route in app.routes if not hasattr(route, "path")
    ]

    assert route_types_missing_path_attr == []

    included_paths = [
        route.path for route in app.routes if type(route).__name__ == "_IncludedRouter"
    ]
    assert included_paths
    assert all(isinstance(path, str) and path.startswith("/") for path in included_paths)


def test_nested_v1_included_routers_expose_prefixed_paths():
    nested_paths = [
        route.path for route in v1_router.routes if type(route).__name__ == "_IncludedRouter"
    ]

    assert nested_paths
    assert all(isinstance(path, str) and path.startswith("/api/v1/") for path in nested_paths)
