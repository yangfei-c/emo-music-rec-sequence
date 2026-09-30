"""Compare uninterrupted training to repeated real checkpoint restoration."""
import argparse
import copy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_config
from emomm_sequence.training.sequence_selector import train, checkpoint_path
from emomm_sequence.training.records import write_json
import torch


def main():
    config, root = load_config("configs/experiments/sequence_routes.yaml")
    args = argparse.Namespace(profile="pilot", variant="set", seed=911, device="cpu", stage="all",
                              limit_tasks=8, max_epochs=1, no_set=False, no_length=False, no_aux_warm=False,
                              objective="frozen", resume=False, run_id="startup_resume_continuous")
    continuous = copy.deepcopy(config)
    continuous["training"].update(microbatch=4, gradient_accumulation_steps=1)
    continuous["profiles"]["pilot"]["max_run_seconds"] = 3600
    if train(continuous, root, args) != 0:
        raise RuntimeError("uninterrupted check did not finish")
    interrupted = copy.deepcopy(continuous)
    interrupted["profiles"]["pilot"]["max_run_seconds"] = 1e-9
    args.run_id = "startup_resume_interrupted"
    exits = []
    for _ in range(6):
        code = train(interrupted, root, args)
        exits.append(code)
        if code == 0:
            break
        if code != 2:
            raise RuntimeError("unexpected interrupted-check error")
        args.resume = True
    else:
        raise RuntimeError("bounded resume check did not terminate")
    states = [torch.load(checkpoint_path(root, "pilot", "set", 911, run) / "last.pt", weights_only=False, map_location="cpu")
              for run in ("startup_resume_continuous", "startup_resume_interrupted")]
    exact = all(torch.equal(value, states[1]["model"][key]) for key, value in states[0]["model"].items())
    write_json(root / "reports" / "checkpoint_resume_check.json", {"status": "passed" if exact else "failed",
               "bitwise_identical_model_tensors": exact, "interrupted_exit_codes": exits,
               "global_steps": [state["global_step"] for state in states], "training_episodes": 8,
               "scope": "actual CPU training, warm+RL, same seed/data, interruption after each batch"})
    if not exact:
        raise AssertionError("checkpoint resume changed model updates")
    print("Bitwise checkpoint-resume equivalence passed")


if __name__ == "__main__":
    main()
