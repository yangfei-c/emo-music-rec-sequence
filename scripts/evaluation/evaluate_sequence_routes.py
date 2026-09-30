"""Evaluate frozen search and learned selectors on the same task manifest."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.evaluation.sequence_benchmark import main
if __name__ == "__main__":
    main()
