from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "promoted_bundle.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_members(members: list[tarfile.TarInfo], version: str) -> None:
    prefix = PurePosixPath("artifacts") / "models" / version
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise RuntimeError(f"Unsafe archive member: {member.name}")
        if path != prefix and prefix not in path.parents:
            raise RuntimeError(f"Unexpected archive member: {member.name}")


def _verify_model_files(model_dir: Path) -> None:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = model_dir / relative
        if not path.is_file():
            raise RuntimeError(f"Promoted artifact is missing {relative}")
        if path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Promoted artifact checksum failed for {relative}")


def restore(manifest_path: Path, archive: Path | None = None) -> Path:
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = release["version"]
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if archive is None:
        if shutil.which("gh") is None:
            raise RuntimeError("GitHub CLI (gh) is required to download the private release asset")
        temporary = tempfile.TemporaryDirectory(prefix="retail-search-artifact-")
        archive = Path(temporary.name) / release["asset_name"]
        subprocess.run(
            [
                "gh",
                "release",
                "download",
                release["release_tag"],
                "--repo",
                release["repository"],
                "--pattern",
                release["asset_name"],
                "--dir",
                temporary.name,
            ],
            check=True,
        )
    archive = archive.resolve()
    if archive.stat().st_size != release["bytes"] or _sha256(archive) != release["sha256"]:
        raise RuntimeError("Promoted bundle size or SHA-256 does not match the tracked release manifest")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        _validate_members(members, version)
        bundle.extractall(ROOT, members=members)
    model_dir = ROOT / "artifacts" / "models" / version
    _verify_model_files(model_dir)
    if temporary is not None:
        temporary.cleanup()
    print(f"Restored and verified promoted model {version} at {model_dir}")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore the frozen promoted model release")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, help="Verify and restore a local release archive")
    args = parser.parse_args()
    restore(args.manifest.resolve(), args.archive)


if __name__ == "__main__":
    main()
