"""Export the current RAG chunks to a timestamped JSON artifact."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Make the project root importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval import ProfileKnowledge  # noqa: E402


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "chunks.json"


def main() -> None:
    knowledge = ProfileKnowledge()
    chunks = knowledge.chunks

    generated_at = datetime.now().astimezone()
    artifact = {
        "metadata": {
            "generated_at": generated_at.isoformat(timespec="seconds"),
            "chunk_count": len(chunks),
            "source_count": len({chunk["source"] for chunk in chunks}),
            "sources": sorted({chunk["source"] for chunk in chunks}),
            "chunking": {
                "strategy": "structure_aware",
                "max_chars": 1200,
                "overlap_chars": 150,
            },
        },
        "chunks": chunks,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"

    # Keep a convenient latest copy and an immutable timestamped snapshot.
    OUTPUT_PATH.write_text(serialized, encoding="utf-8")
    timestamp = generated_at.strftime("%Y%m%d_%H%M%S_%z")
    snapshot_path = ARTIFACTS_DIR / f"chunks_{timestamp}.json"
    snapshot_path.write_text(serialized, encoding="utf-8")

    print(f"Exported {len(chunks)} chunks to: {OUTPUT_PATH}")
    print(f"Timestamped snapshot: {snapshot_path}")
    print(f"Generated at: {artifact['metadata']['generated_at']}")


if __name__ == "__main__":
    main()
