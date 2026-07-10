from app.api.v1 import v1_router
from app.main import app


def test_app_routes_expose_path_for_included_routers():
    missing_path_route_types = [
        type(route).__name__ for route in app.routes if not hasattr(route, "path")
    ]

    assert missing_path_route_types == []

    included_paths = {
        route.path for route in app.routes if type(route).__name__ == "_IncludedRouter"
    }
    assert {"/api/v1", "/api/auth", "/api/portfolio", "/api/cron"} <= included_paths


def test_nested_v1_included_routers_expose_prefixed_paths():
    nested_paths = {
        route.path for route in v1_router.routes if type(route).__name__ == "_IncludedRouter"
    }

    assert "/api/v1/auth" in nested_paths
    assert "/api/v1/portfolio" in nested_paths
