from pathlib import Path
import re


class LocalObjectStorage:
    def __init__(self, root: Path):
        self.root = root

    def put_bytes(self, key: str, content: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def get_bytes(self, key: str) -> bytes:
        return (self.root / key).read_bytes()


class MinioObjectStorage:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        from minio import Minio

        self.bucket = bucket
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put_bytes(self, key: str, content: bytes) -> None:
        from io import BytesIO

        self.client.put_object(
            self.bucket,
            key,
            BytesIO(content),
            length=len(content),
        )

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


def object_key_for(checksum: str, filename: str) -> str:
    safe_name = re.sub(r"\s+", "_", filename.strip())
    safe_name = safe_name.replace("/", "_").replace("\\", "_")
    return f"sources/{checksum}/{safe_name}"
