from app.services.object_storage import LocalObjectStorage, object_key_for


def test_object_key_uses_checksum_and_safe_filename():
    key = object_key_for(checksum="abc123", filename="资料 01.pdf")

    assert key == "sources/abc123/资料_01.pdf"


def test_local_object_storage_round_trips_bytes(tmp_path):
    storage = LocalObjectStorage(root=tmp_path)
    key = object_key_for(checksum="abc123", filename="资料.txt")

    storage.put_bytes(key, b"hello")

    assert storage.get_bytes(key) == b"hello"
