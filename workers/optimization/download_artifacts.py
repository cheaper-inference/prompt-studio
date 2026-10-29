"""Explicit setup step. No model download occurs during a request."""
import json
import os
import re
from pathlib import Path
from operations import configured_operations

ROOT = Path(__file__).parent


def download_artifacts(root: Path = ROOT) -> None:
    manifest = json.loads((root / "artifacts.json").read_text())
    if not isinstance(manifest, dict):
        raise RuntimeError("Artifact manifest must be an object keyed by operation")
    operations = configured_operations(manifest)

    # Validate the entire selection before importing download clients or doing
    # any network work, even when an invalid operation appears last.
    for key in operations:
        artifact = manifest[key]
        if (not isinstance(artifact, dict)
                or not isinstance(artifact.get("model"), str)
                or not artifact["model"].strip()
                or not isinstance(artifact.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", artifact["revision"])):
            raise RuntimeError(f"Artifact {key!r} requires a model and a pinned 40-character revision")

    if not operations:
        return
    from huggingface_hub import snapshot_download

    os.environ["TIKTOKEN_CACHE_DIR"] = str(root / "models" / "tiktoken")
    for key in operations:
        artifact = manifest[key]
        snapshot_download(artifact["model"], revision=artifact["revision"], local_dir=root / "models" / key)

    if "llmlingua2" in operations:
        import tiktoken
        tiktoken.get_encoding("cl100k_base")


if __name__ == "__main__":
    download_artifacts()
