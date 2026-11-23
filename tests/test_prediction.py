import unittest
from io import BytesIO
from email.message import Message
from unittest.mock import patch

from fastapi.testclient import TestClient
from scipy.sparse import csr_matrix

from api.predict import handler as VercelHandler
from preprocess import prepare_data
from server import app


client = TestClient(app)


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_stems_before_vectorizing(self) -> None:
        expected = csr_matrix([[1]])
        with patch("preprocess.cvectorizer.transform", return_value=expected) as transform:
            result = prepare_data("https://secure-login.example.com")

        self.assertIs(result, expected)
        transform.assert_called_once_with(["https secur login exampl com"])


class PredictionApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_predict_preserves_response_shape(self) -> None:
        with patch("server.predictor", return_value=1):
            response = client.post(
                "/api/predict",
                json={"url": "https://suspicious.example/path"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"label": "bad"})

    def test_local_legacy_route_remains_available(self) -> None:
        with patch("server.predictor", return_value=0):
            response = client.post(
                "/predict",
                json={"url": "https://example.com"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"label": "good"})

    def test_invalid_schemes_are_rejected(self) -> None:
        response = client.post("/predict", json={"url": "javascript:alert(1)"})
        self.assertEqual(response.status_code, 422)

    def test_url_length_is_bounded(self) -> None:
        response = client.post(
            "/predict",
            json={"url": f"https://example.com/{'a' * 2048}"},
        )
        self.assertEqual(response.status_code, 422)


class VercelFunctionTests(unittest.TestCase):
    def test_valid_request_returns_prediction(self) -> None:
        request = object.__new__(VercelHandler)
        request.path = "/api/predict"
        request.headers = Message()
        body = b'{"url":"https://suspicious.example"}'
        request.headers["Content-Length"] = str(len(body))
        request.rfile = BytesIO(body)
        request.wfile = BytesIO()
        request.send_response = lambda status: setattr(request, "status", status)
        request.send_header = lambda *_: None
        request.end_headers = lambda: None

        with patch("api.predict.predictor", return_value=1):
            VercelHandler.do_POST(request)

        self.assertEqual(request.status, 200)
        self.assertEqual(
            request.wfile.getvalue(),
            b'{"label": "bad"}',
        )


if __name__ == "__main__":
    unittest.main()
