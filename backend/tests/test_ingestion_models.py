from app.models.ingestion import SourceManifest


def test_source_manifest_tracks_uploaded_document_status():
    manifest = SourceManifest(
        source_id="source:uploaded:abc",
        filename="资料.pdf",
        mime_type="application/pdf",
        checksum="abc123",
        status="registered",
    )

    assert manifest.source_id == "source:uploaded:abc"
    assert manifest.status == "registered"
