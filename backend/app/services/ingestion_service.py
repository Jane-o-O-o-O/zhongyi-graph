from app.models.ingestion import IngestionJob, SourceManifest


class IngestionService:
    def __init__(self):
        self.sources: dict[str, SourceManifest] = {}
        self.jobs: dict[str, IngestionJob] = {}

    def register_source(self, manifest: SourceManifest) -> SourceManifest:
        self.sources[manifest.source_id] = manifest
        return manifest

    def create_job(self, source_ids: list[str]) -> IngestionJob:
        job = IngestionJob(job_id=f"job:{len(self.jobs) + 1}", source_ids=source_ids)
        self.jobs[job.job_id] = job
        return job
