from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.services.model_clients import EmbeddingClient, RerankClient
from app.services.ragflow_compat.doc_store import RagflowDocStore
from app.services.ragflow_compat.fulltext import RagflowFulltextRetriever
from app.services.ragflow_compat.repository import RagflowRetrievalRepository
from app.services.ragflow_compat.schemas import RetrievalChunk, RetrievalDocument


def test_fulltext_retriever_uses_query_tokens_doc_store_and_rerank():
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
                chunk_id="chunk:1",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:1",
                chunk_order_int=1,
                page_num_int=1,
                title="不寐",
                section_path=[],
                content="失眠可辨为心脾两虚，常用归脾汤。",
                content_with_weight="不寐 失眠 心脾两虚 归脾汤 失眠可辨为心脾两虚，常用归脾汤。",
                content_ltks="失眠 心脾两虚 归脾汤",
                token_count=18,
            ),
            RetrievalChunk(
                chunk_id="chunk:2",
                doc_id="doc",
                source_id="doc",
                parent_unit_id="unit:2",
                chunk_order_int=2,
                page_num_int=1,
                title="少阳",
                section_path=[],
                content="柴胡桂枝干姜汤用于往来寒热。",
                content_with_weight="少阳 柴胡桂枝干姜汤 往来寒热",
                content_ltks="柴胡桂枝干姜汤 往来寒热",
                token_count=18,
            ),
        ]
    )
    retriever = RagflowFulltextRetriever(
        doc_store=RagflowDocStore(repository, EmbeddingClient.demo()),
        rerank_client=RerankClient.demo(),
    )

    result = retriever.retrieve("睡不着从哪些证候分析？", top_k=1)

    assert result.keywords[:2] == ["失眠", "证候"]
    assert len(result.hits) == 1
    assert result.hits[0].chunk.chunk_id == "chunk:1"
    assert result.hits[0].rerank_score > 0
