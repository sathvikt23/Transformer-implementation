import os
import shutil
import tempfile
from src.tokenizer import get_global_tokenizer
from src.shards import create_synthetic_shards
from src.eval_firewall import EvalFirewall
from src.mixture import MixtureScheduler
from src.opus import OPUSEngine
from src.packer import SequencePacker
from src.batch import BatchBuilder
from src.ledger import ConsumptionLedger, LearningLedger
from src.checkpoint import CheckpointManager
from src.train_loop import TrainingExecutionEngine


def test_resume_no_skip_no_repeat():
    tmp_dir = tempfile.mkdtemp(prefix="v5_test_resume_")
    try:
        tokenizer = get_global_tokenizer()
        eval_firewall = EvalFirewall()
        mixture_scheduler = MixtureScheduler("reasoning-heavy-midtrain")
        opus_engine = OPUSEngine()
        packer = SequencePacker(128)
        batch_builder = BatchBuilder(128)
        shards = create_synthetic_shards(tokenizer, count=5)

        cons_ledger = ConsumptionLedger(os.path.join(tmp_dir, "ledgers", "consumption.db"))
        learn_ledger = LearningLedger(os.path.join(tmp_dir, "ledgers", "learning.db"))
        ckpt_mgr = CheckpointManager(os.path.join(tmp_dir, "checkpoints"))

        engine = TrainingExecutionEngine(
            artifacts_dir=tmp_dir,
            tokenizer=tokenizer,
            eval_firewall=eval_firewall,
            mixture_scheduler=mixture_scheduler,
            opus_engine=opus_engine,
            packer=packer,
            batch_builder=batch_builder,
            consumption_ledger=cons_ledger,
            learning_ledger=learn_ledger,
            checkpoint_manager=ckpt_mgr,
            logger_func=None,
        )

        # Run steps 1..5
        mbs = []
        for step in range(1, 6):
            lane = mixture_scheduler.select_lane_for_step(step)
            mb, row_id = engine.run_training_step(step, 1, shards[0].documents, lane, "reasoning-heavy-midtrain")
            mbs.append(mb)

        # Checkpoint saved at step 3
        ckpt_mgr.save_checkpoint(
            global_step=3,
            model_state={"dummy": 1},
            optimizer_state={"dummy": 1},
            scheduler_state={"dummy": 1},
            rng_state={"dummy": 1},
            dataloader_state={"step": 3},
            ledger_offset=3,
        )

        # Simulate Crash after step 5
        # Resume from checkpoint step 3
        ckpt = ckpt_mgr.load_checkpoint(3)
        assert ckpt["ledger_offset"] == 3

        resumed_step = 4
        resumed_lane = mixture_scheduler.select_lane_for_step(resumed_step)
        resumed_mb, _ = engine.run_training_step(
            resumed_step, 1, shards[0].documents, resumed_lane, "reasoning-heavy-midtrain", branch_id="br_resumed"
        )

        # The batch produced at step 4 post-resume MUST match the exact batch ID of original step 4
        original_step4_mb_id = mbs[3].microbatch_id  # index 3 is step 4
        resumed_step4_mb_id = resumed_mb.microbatch_id

        assert resumed_step4_mb_id == original_step4_mb_id, f"Resume sequence mismatch! Expected {original_step4_mb_id}, got {resumed_step4_mb_id}"

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
