from __future__ import annotations

from pathlib import Path
import sys

PROJECT_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(PROJECT_BACKEND) not in sys.path:
    sys.path.insert(0, str(PROJECT_BACKEND))


def format_audit(audit: dict) -> str:
    readiness = audit.pop("readiness", None)
    parts = [f"{key}={value}" for key, value in sorted(audit.items())]
    if readiness:
        parts.extend(_format_readiness(readiness))
    return " ".join(parts)


def main() -> None:
    from app.core.config import get_settings
    from app.services.ingestion_repository import IngestionRepository
    from app.services.ragflow_compat.repository import RagflowRetrievalRepository

    settings = get_settings()
    ingestion_repository = IngestionRepository.from_dsn(settings.postgres_dsn)
    repository = RagflowRetrievalRepository(ingestion_repository.engine)
    audit = repository.audit().__dict__
    audit["readiness"] = repository.readiness()
    print(format_audit(audit))


def _format_readiness(readiness: dict) -> list[str]:
    parts = [f"ready={readiness.get('ready')}"]
    blockers = readiness.get("blockers") or []
    warnings = readiness.get("warnings") or []
    parts.append("blockers=" + (",".join(blockers) if blockers else "none"))
    parts.append("warnings=" + (",".join(warnings) if warnings else "none"))
    for name, coverage in sorted((readiness.get("vector_coverage") or {}).items()):
        parts.append(f"{name}_vector_ratio={coverage.get('ratio', 0)}")
        parts.append(f"{name}_vectors={coverage.get('embedded', 0)}/{coverage.get('total', 0)}")
    for name, coverage in sorted((readiness.get("evidence_coverage") or {}).items()):
        parts.append(f"{name}_evidence_ratio={coverage.get('ratio', 0)}")
        parts.append(
            f"{name}_evidence={coverage.get('with_evidence', 0)}/{coverage.get('total', 0)}"
        )
    for bucket, count in sorted((readiness.get("chunk_token_buckets") or {}).items()):
        parts.append(f"bucket_{bucket}={count}")
    sync_plan = readiness.get("vector_sync_plan") or {}
    if sync_plan:
        parts.append(f"sync_limit={sync_plan.get('balanced_limit', 0)}")
        parts.append(f"sync_command='{sync_plan.get('recommended_command', '')}'")
    return parts


if __name__ == "__main__":
    main()
