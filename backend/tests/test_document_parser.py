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

    pages, units, chunks, status = parser.parse(
        source,
        "失眠可辨为心脾两虚。\n治以补益心脾。".encode("utf-8"),
    )

    assert status == "parsed"
    assert pages[0].text.startswith("失眠")
    assert units[0].content == "失眠可辨为心脾两虚。\n治以补益心脾。"
    assert chunks[0].content == "失眠可辨为心脾两虚。\n治以补益心脾。"
    assert chunks[0].source_id == source.source_id
    assert chunks[0].parent_unit_id == units[0].unit_id


def test_document_parser_creates_table_chunk_for_csv():
    parser = DocumentParser()
    source = SourceManifest(
        source_id="source:uploaded:csv",
        filename="方剂.csv",
        mime_type="text/csv",
        checksum="abc123",
        status="registered",
    )

    _pages, units, chunks, status = parser.parse(source, "方剂,药味\n归脾汤,党参".encode("utf-8"))

    assert status == "parsed"
    assert units[0].unit_type == "table"
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

    _pages, units, chunks, status = parser.parse(source, b"fake-image")

    assert status == "parsed"
    assert units[0].unit_type == "ocr"
    assert chunks[0].content_type == "ocr"
    assert "OCR识别文本" in chunks[0].content


def test_document_parser_builds_extraction_units_from_real_tcm_markers():
    parser = DocumentParser(parent_target_chars=160, parent_max_chars=240, child_target_chars=70)
    source = SourceManifest(
        source_id="source:uploaded:tcm",
        filename="普济方.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="registered",
    )
    text = """
<篇名>普济方
书名：普济方

<目录>卷一\\方脉总论

<篇名>五常大论

属性：头者诸阳之会。是以头痛多属于阳也。独厥阴肝脉，上入颃颡，连目系，上出额，与督脉会于颠。
又有病头痛连齿，时发时止，数岁不已者。此肾所生病也。

<目录>卷一\\方脉总论

<篇名>五脏像位

属性：肝名龙烟。于五行为木。肾名玄冥。于五行为水。
""".strip()

    _pages, units, chunks, status = parser.parse(source, text.encode("utf-8"))

    assert status == "parsed"
    assert [unit.title for unit in units][:2] == ["五常大论", "五脏像位"]
    assert units[0].section_path == ["普济方", "卷一\\方脉总论", "五常大论"]
    assert units[0].unit_type == "article"
    assert "头痛多属于阳" in units[0].content
    assert "肝名龙烟" not in units[0].content
    assert all(chunk.parent_unit_id for chunk in chunks)
    assert {chunk.metadata["section_path"][-1] for chunk in chunks} == {"五常大论", "五脏像位"}


def test_document_parser_splits_long_parent_units_on_sentence_boundaries():
    parser = DocumentParser(parent_target_chars=70, parent_max_chars=110, child_target_chars=45)
    source = SourceManifest(
        source_id="source:uploaded:casebook",
        filename="王氏医案绎注.txt",
        mime_type="text/plain",
        checksum="abc123",
        status="registered",
    )
    text = """
<篇名>王氏医案绎注

<目录>

<篇名>卷一

属性：夏令某登厕。忽然体冷汗出。气怯神疲。孟英视之曰阳气欲脱也。急煎姜汤灌之即安。范庆簪年逾五十。素患痰嗽。骤然吐血。孟英诊曰气虚而血无统摄也。乃以参术苓草为方。五帖而安。
""".strip()

    _pages, units, chunks, _status = parser.parse(source, text.encode("utf-8"))

    assert len(units) > 1
    assert all(unit.content.endswith(("。", "也。", "安。")) for unit in units)
    assert all(not unit.content.startswith("忽然") for unit in units[1:])
    assert all(chunk.metadata["parent_unit_index"] == chunk.unit_index for chunk in chunks)
