from pathlib import Path

from scripts.import_literature_graphrag import (
    ImportProgress,
    classify_file,
    iter_importable_files,
    should_publish_run,
)


def test_iter_importable_files_skips_unsupported_hidden_and_large_json(tmp_path):
    (tmp_path / "book.md").write_text("归脾汤", encoding="utf-8")
    (tmp_path / "table.csv").write_text("name,value\n党参,补气", encoding="utf-8")
    (tmp_path / "legacy.doc").write_bytes(b"legacy")
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / "._book.md").write_bytes(b"")
    large_json = tmp_path / "dialogue.json"
    large_json.write_bytes(b"{" + (b" " * 101) + b"}")

    files = list(iter_importable_files(tmp_path, max_json_bytes=100, include_images=False))

    assert files == [tmp_path / "book.md", tmp_path / "table.csv"]
    assert classify_file(large_json, max_json_bytes=100, include_images=False).reason == "large_json"
    assert classify_file(tmp_path / "legacy.doc", max_json_bytes=100).reason == "unsupported"


def test_import_progress_treats_failed_rows_as_retryable(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    progress_path.write_text(
        '{"path": "/data/a.md", "status": "failed"}\n'
        '{"path": "/data/b.md", "status": "parsed"}\n'
        '{"path": "/data/c.md", "status": "published"}\n',
        encoding="utf-8",
    )

    progress = ImportProgress(progress_path)

    assert progress.completed_paths == {"/data/b.md", "/data/c.md"}
    assert "/data/a.md" not in progress.completed_paths


def test_should_publish_only_when_relations_exist_unless_forced():
    assert should_publish_run({"relation_count": 1}, publish_all=False)
    assert not should_publish_run({"relation_count": 0}, publish_all=False)
    assert should_publish_run({"relation_count": 0}, publish_all=True)
