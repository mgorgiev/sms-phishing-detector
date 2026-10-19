from fastapi.testclient import TestClient

from app.main import create_app


class FakeClassifier:
    def __init__(self):
        self.calls: list[list[str]] = []

    def predict(self, texts):
        self.calls.append(texts)
        return [("smishing", {"ham": 0.01, "spam": 0.04, "smishing": 0.95}) for _ in texts]


def client(fake: FakeClassifier) -> TestClient:
    return TestClient(create_app(fake))


def test_predict_returns_label_and_scores():
    fake = FakeClassifier()
    with client(fake) as c:
        resp = c.post("/predict", json={"text": "Your account is locked, verify at bit.do/x"})

    assert resp.status_code == 200
    assert resp.json() == {
        "label": "smishing",
        "scores": {"ham": 0.01, "spam": 0.04, "smishing": 0.95},
    }
    assert fake.calls == [["Your account is locked, verify at bit.do/x"]]


def test_predict_rejects_empty_and_missing_text():
    with client(FakeClassifier()) as c:
        assert c.post("/predict", json={"text": ""}).status_code == 422
        assert c.post("/predict", json={}).status_code == 422
        assert c.post("/predict", json={"text": "a" * 5001}).status_code == 422


def test_healthz():
    with client(FakeClassifier()) as c:
        assert c.get("/healthz").json() == {"status": "ok"}
