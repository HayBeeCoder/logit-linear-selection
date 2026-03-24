import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

class CheckpointManager:
    """
    Chunk-level checkpointing for Phase 1.

    Files:
      - phase1_progress.json  (human-readable status)
      - phase1.pkl            (resume data; pickled, trusted-local only)

    Notes:
      - Uses atomic writes (temp file + os.replace) to avoid corruption on crash.
      - WARNING: pickle is unsafe to load from untrusted locations.
    """
    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.dir / "phase1_progress.json"
        self.data_path = self.dir / "phase1.pkl"

    def _atomic_write_text(self, path: Path, text: str):
        with tempfile.NamedTemporaryFile("w", dir=str(self.dir), delete=False, encoding="utf-8") as tmp:
            tmp.write(text)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)

    def _atomic_write_pickle(self, path: Path, obj: Any):
        with tempfile.NamedTemporaryFile("wb", dir=str(self.dir), delete=False) as tmp:
            pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, path)

    def save(
        self,
        *,
        chunk_start: int,
        chunk_end: int,
        total_rank_examples: int,
        local_tuples: Any,
        extra: Optional[Dict[str, Any]] = None,
    ):
        if not hasattr(local_tuples, "__len__"):
            raise TypeError("local_tuples must be a sized collection (e.g., list).")

        progress: Dict[str, Any] = {
            "chunk_start": int(chunk_start),
            "chunk_end": int(chunk_end),
            "total_rank_examples": int(total_rank_examples),
            "local_tuples_count": int(len(local_tuples)),
            "pct_rank_complete": (100.0 * float(chunk_end) / float(max(total_rank_examples, 1))),
        }
        if extra:
            progress.update(extra)

        # Write progress JSON first (human-readable), then pickle.
        # Both are atomic; worst case they are temporarily out of sync between the two writes.
        self._atomic_write_text(self.progress_path, json.dumps(progress, indent=2))

        self._atomic_write_pickle(
            self.data_path,
            {
                "resume_chunk_idx": int(chunk_end),  # next loop starts here
                "local_tuples": local_tuples,
            },
        )

        print(
            f"[checkpoint] saved rank_progress={chunk_end}/{total_rank_examples} "
            f"({progress['pct_rank_complete']:.2f}%) tuples={len(local_tuples)} dir={self.dir}"
        )

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Loads resume data (pickle). Also attempts to load progress JSON and attach it.
        If JSON is missing/corrupt but pickle exists, still resumes.
        """
        if not self.data_path.exists():
            return None

        with open(self.data_path, "rb") as f:
            data = pickle.load(f)

        progress = None
        if self.progress_path.exists():
            try:
                progress = json.loads(self.progress_path.read_text(encoding="utf-8"))
            except Exception:
                progress = None

        if progress is not None:
            data["_progress"] = progress
        return data

    def clear(self):
        # Best-effort cleanup; leaving one file behind is not catastrophic.
        for p in (self.progress_path, self.data_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
        print(f"[checkpoint] cleared {self.dir}")