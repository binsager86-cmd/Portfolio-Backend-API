import sys
from types import ModuleType, SimpleNamespace


def test_registers_android_fcm_token(test_client, auth_headers):
    response = test_client.post(
        "/api/v1/notifications/register-token",
        headers=auth_headers,
        json={
            "token": "fcm-registration-token-for-smoke-test",
            "platform": "android",
            "token_provider": "fcm",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_builds_fcm_notification_with_channel_and_string_data(monkeypatch, tmp_path):
    from app.services import push_service

    service_account = tmp_path / "service-account.json"
    service_account.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        push_service.get_settings(), "FIREBASE_SERVICE_ACCOUNT_FILE", str(service_account)
    )

    sent_messages = []

    class FakeMessaging:
        class Notification:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class AndroidNotification:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class AndroidConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Message:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def send(message):
            sent_messages.append(message)

    fake_admin = ModuleType("firebase_admin")
    fake_admin.get_app = lambda: (_ for _ in ()).throw(ValueError())
    fake_admin.initialize_app = lambda credential: None
    fake_credentials = ModuleType("firebase_admin.credentials")
    fake_credentials.Certificate = lambda path: path
    fake_admin.credentials = fake_credentials
    fake_admin.messaging = FakeMessaging
    monkeypatch.setitem(sys.modules, "firebase_admin", fake_admin)
    monkeypatch.setitem(sys.modules, "firebase_admin.credentials", fake_credentials)
    monkeypatch.setitem(sys.modules, "firebase_admin.messaging", FakeMessaging)

    assert push_service._send_fcm_notification(
        "fcm-registration-token-for-smoke-test",
        "Portfolio update",
        "Your portfolio moved.",
        {"type": "portfolio_update", "changePct": 2.5},
        "portfolio-updates",
    )

    message = sent_messages[0].kwargs
    assert message["token"] == "fcm-registration-token-for-smoke-test"
    assert message["data"] == {"type": "portfolio_update", "changePct": "2.5"}
    assert message["android"].kwargs["notification"].kwargs["channel_id"] == "portfolio-updates"