"""Prepare immutable sequence tasks and label-separated feature banks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import main
if __name__ == "__main__":
    main()
