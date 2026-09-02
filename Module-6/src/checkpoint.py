import os
import json
from typing import Dict, Any, Tuple, List

REQUIRED_CHECKPOINT_FIELDS = [
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "rng_state",
    "dataloader_state",
    "ledger_offset",
]


class CheckpointManager:
    """
    Checkpoint Manager enforcing state preservation across 6 required fields.
    Crucially ties training state to ledger_offset for provable resume & replay.
    """

    def __init__(self, checkpoints_dir: str):
        self.checkpoints_dir = checkpoints_dir
        os.makedirs(checkpoints_dir, exist_ok=True)

    def save_checkpoint(
        self,
        global_step: int,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        scheduler_state: Dict[str, Any],
        rng_state: Dict[str, Any],
        dataloader_state: Dict[str, Any],
        ledger_offset: int,
    ) -> str:
        step_dir = os.path.join(self.checkpoints_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)

        ckpt_payload = {
            "checkpoint_id": f"step_{global_step}",
            "branch_id": "br_main",
            "run_id": "run_v5_01",
            "global_step": global_step,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "rng_state": rng_state,
            "dataloader_state": dataloader_state,
            "ledger_offset": ledger_offset,
        }

        # Verify completeness before writing
        is_complete, missing = verify_checkpoint_completeness(ckpt_payload)
        if not is_complete:
            raise ValueError(f"Cannot save incomplete checkpoint! Missing fields: {missing}")

        file_path = os.path.join(step_dir, "checkpoint.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(ckpt_payload, f, indent=2)

        return file_path

    def load_checkpoint(self, global_step: int) -> Dict[str, Any]:
        file_path = os.path.join(self.checkpoints_dir, f"step_{global_step}", "checkpoint.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Checkpoint for step {global_step} not found at {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            ckpt_payload = json.load(f)

        is_complete, missing = verify_checkpoint_completeness(ckpt_payload)
        if not is_complete:
            raise ValueError(f"Loaded checkpoint is incomplete! Missing fields: {missing}")

        return ckpt_payload


def verify_checkpoint_completeness(ckpt: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verifies that a checkpoint dictionary contains all 6 required fields.
    Returns (is_complete, list_of_missing_fields).
    """
    missing = [field for field in REQUIRED_CHECKPOINT_FIELDS if field not in ckpt]
    return (len(missing) == 0, missing)
