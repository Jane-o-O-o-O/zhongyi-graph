from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.ragflow_compat.context import expand_parent_context
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalDocument


def test_expand_parent_context_returns_neighboring_chunks_from_same_parent_unit():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    repository = RagflowRetrievalRepository(engine)
    repository.replace_documents(
        [RetrievalDocument(doc_id="doc", source_id="doc", filename="doc.txt")]
    )
    repository.replace_chunks(
        [
            RetrievalChunk(
                chunk_id=f"chunk:{index}",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=index,
                page_num_int=1,
                title="",
                section_path=[],
                content=f"内容 {index}",
                content_with_weight=f"内容 {index}",
                token_count=10,
            )
            for index in range(1, 4)
        ]
    )

    expanded = expand_parent_context(repository, ["chunk:2"], window=1)

    assert [chunk.chunk_id for chunk in expanded] == ["chunk:1", "chunk:2", "chunk:3"]
