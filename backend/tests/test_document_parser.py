from app.models.ingestion import SourceManifest
from app.services.document_parser import DocumentParser


class FakeOcrClient:
    def recognize_image(self, content: bytes, mime_type: str) -> str:
        return "OCR识别文本：失眠可辨为心脾两虚。"


def test_document_parser_chunks_plain_text():
    parser = DocumentParser()
    source = SourceManifest(
        source_id="source:uploaded:txt",
        filename="资料.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="registered",
    )

    pages, chunks, status = parser.parse(source, "失眠可辨为心脾两虚。\n治以补益心脾。".encode("utf-8"))

    assert status == "parsed"
    assert pages[0].text.startswith("失眠")
    assert chunks[0].content == "失眠可辨为心脾两虚。\n治以补益心脾。"
    assert chunks[0].source_id == source.source_id


def test_document_parser_creates_table_chunk_for_csv():
    parser = DocumentParser()
    source = SourceManifest(
        source_id="source:uploaded:csv",
        filename="方剂.csv",
        mime_type="text/csv",
        checksum="abc123",
        status="registered",
    )

    _pages, chunks, status = parser.parse(source, "方剂,药味\n归脾汤,党参".encode("utf-8"))

    assert status == "parsed"
    assert chunks[0].content_type == "table"
    assert "归脾汤" in chunks[0].content


def test_document_parser_uses_ocr_for_image_sources():
    parser = DocumentParser(ocr_client=FakeOcrClient())
    source = SourceManifest(
        source_id="source:uploaded:image",
        filename="扫描.png",
        mime_type="image/png",
        checksum="abc123",
        status="registered",
    )

    _pages, chunks, status = parser.parse(source, b"fake-image")

    assert status == "parsed"
    assert chunks[0].content_type == "ocr"
    assert "OCR识别文本" in chunks[0].content
