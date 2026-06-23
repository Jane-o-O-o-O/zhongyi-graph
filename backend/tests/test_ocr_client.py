import json

import httpx

from app.services.ocr_client import OcrClient


def test_ocr_client_calls_siliconflow_chat_model_with_image_payload():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["json"] = json.loads(request.read().decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "识别出的中医文本"}}]},
        )

    client = OcrClient(
        base_url="https://api.siliconflow.cn/v1",
        api_key="secret",
        model="deepseek-ai/DeepSeek-OCR",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    text = client.recognize_image(b"fake-image", mime_type="image/png")

    assert text == "识别出的中医文本"
    assert captured["url"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret"
    assert captured["json"]["model"] == "deepseek-ai/DeepSeek-OCR"
    assert captured["json"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
