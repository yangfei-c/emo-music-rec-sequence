"""Train the candidate-set or direct-path policy using frozen features."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.training.sequence_selector import main
if __name__ == "__main__":
    main()
