import httpx

from app.services.model_clients import StructuredExtractionClient


class CommunityReportHttpClient:
    def __init__(self):
        self.requests = []

    def post(self, url: str, headers: dict, json: dict) -> httpx.Response:
        self.requests.append(json)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"title":"白芍养血社区",'
                                '"summary":"围绕白芍和养血敛阴形成的功效社区。",'
                                '"findings":[{"summary":"功效主题","explanation":"白芍连接养血敛阴。"}],'
                                '"rating":1.0,'
                                '"rating_explanation":"结构清晰"}'
                            )
                        }
                    }
                ]
            },
        )


def test_structured_extraction_client_generates_community_report_from_llm_json():
    http_client = CommunityReportHttpClient()
    client = StructuredExtractionClient(
        base_url="http://localhost:8088/v1",
        api_key="demo",
        model="demo-extraction",
        http_client=http_client,
    )

    report = client.generate_community_report(
        community_id=0,
        entities=["白芍", "养血敛阴"],
        relations=["白芍 功效 养血敛阴"],
    )

    assert report["title"] == "白芍养血社区"
    assert report["summary"] == "围绕白芍和养血敛阴形成的功效社区。"
    assert report["findings"][0]["summary"] == "功效主题"
    payload = http_client.requests[0]
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert "白芍 功效 养血敛阴" in payload["messages"][1]["content"]
