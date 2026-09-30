"""Artifact storage. The DB only ever stores *relative* artifact URIs (never blobs).

Layout under the artifact root (a volume shared with titiler, or an S3/MinIO mount):

    species/<taxon_key>/features_<taxon_key>.parquet
    species/<taxon_key>/archive/features_<taxon_key>.parquet   (inactive species, zstd)
    species/<taxon_key>/v<version>/model.pkl
    species/<taxon_key>/v<version>/{suitability,mess,zones}.tif
    species/<taxon_key>/v<version>/suitability_<scenario>.tif
    species/<taxon_key>/v<version>/bulletin.{html,pdf}
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import quote


class ArtifactStore:
    def __init__(self, root: Path, titiler_root: str) -> None:
        self.root = root
        self.titiler_root = titiler_root.rstrip("/")

    def species_dir(self, taxon_key: int) -> PurePosixPath:
        return PurePosixPath("species") / str(taxon_key)

    def version_dir(self, taxon_key: int, version: int) -> PurePosixPath:
        return self.species_dir(taxon_key) / f"v{version}"

    def features_uri(self, taxon_key: int) -> str:
        return str(self.species_dir(taxon_key) / f"features_{taxon_key}.parquet")

    def archived_features_uri(self, taxon_key: int) -> str:
        """zstd-recompressed feature table of an inactive species (orchestration/maintenance)."""
        return str(self.species_dir(taxon_key) / "archive" / f"features_{taxon_key}.parquet")

    def local_path(self, uri: str) -> Path:
        rel = PurePosixPath(uri)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Artifact URI must be relative: {uri}")
        return self.root.joinpath(*rel.parts)

    def relative_uri(self, path: Path) -> str:
        return PurePosixPath(path.resolve().relative_to(self.root.resolve()).as_posix()).as_posix()

    def titiler_url(self, uri: str) -> str:
        """Dataset URL as seen from inside the titiler container (URL-encoded)."""
        return quote(f"{self.titiler_root}/{uri}", safe="")
