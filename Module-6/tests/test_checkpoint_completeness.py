import os
import shutil
import tempfile
import pytest
from src.checkpoint import CheckpointManager, verify_checkpoint_completeness, REQUIRED_CHECKPOINT_FIELDS


def test_checkpoint_completeness_all_six_fields():
    tmp_dir = tempfile.mkdtemp(prefix="v5_test_ckpt_")
    try:
        mgr = CheckpointManager(tmp_dir)

        # Valid payload with all 6 required fields
        model_s = {"weight": 1.0}
        optim_s = {"step": 100}
        sched_s = {"lr": 0.001}
        rng_s = {"seed": 42}
        dloader_s = {"idx": 500}
        ledger_offset = 12345

        ckpt_file = mgr.save_checkpoint(
            global_step=100,
            model_state=model_s,
            optimizer_state=optim_s,
            scheduler_state=sched_s,
            rng_state=rng_s,
            dataloader_state=dloader_s,
            ledger_offset=ledger_offset,
        )

        assert os.path.exists(ckpt_file)

        loaded = mgr.load_checkpoint(100)
        is_complete, missing = verify_checkpoint_completeness(loaded)
        assert is_complete is True
        assert len(missing) == 0
        assert loaded["ledger_offset"] == ledger_offset

        # Test incomplete payload raises error
        incomplete_dict = {
            "global_step": 100,
            "model_state": model_s,
            "optimizer_state": optim_s,
            # Missing scheduler_state, rng_state, dataloader_state, ledger_offset
        }
        complete, miss = verify_checkpoint_completeness(incomplete_dict)
        assert complete is False
        assert set(miss) == {"scheduler_state", "rng_state", "dataloader_state", "ledger_offset"}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
