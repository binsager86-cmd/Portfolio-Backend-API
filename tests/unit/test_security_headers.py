def test_docs_route_csp_allows_swagger_assets(test_client):
    response = test_client.get("/docs")

    assert response.status_code == 200
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in csp
    assert "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in csp
    assert "img-src 'self' data: https://fastapi.tiangolo.com" in csp
    assert "frame-ancestors 'none'" in csp


def test_redoc_route_csp_allows_docs_assets(test_client):
    response = test_client.get("/redoc")

    assert response.status_code == 200
    csp = response.headers["Content-Security-Policy"]
    assert "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in csp
    assert "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net" in csp
    assert "frame-ancestors 'none'" in csp


def test_api_route_csp_remains_strict(test_client):
    response = test_client.get("/health")

    assert response.status_code == 200
    assert response.headers["Content-Security-Policy"] == "default-src 'self'; frame-ancestors 'none'"
