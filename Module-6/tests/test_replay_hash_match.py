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


def test_replay_hash_match():
    tmp_dir = tempfile.mkdtemp(prefix="v5_test_replay_")
    try:
        tokenizer = get_global_tokenizer()
        eval_firewall = EvalFirewall()
        mixture_scheduler = MixtureScheduler("reasoning-heavy-midtrain")
        opus_engine = OPUSEngine()
        packer = SequencePacker(128)
        batch_builder = BatchBuilder(128)
        shards = create_synthetic_shards(tokenizer, count=3)

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

        row_ids = []
        for step in range(10, 15):
            lane = mixture_scheduler.select_lane_for_step(step)
            mb, row_id = engine.run_training_step(step, 1, shards[0].documents, lane, "reasoning-heavy-midtrain")
            row_ids.append(row_id)

        min_row = min(row_ids)
        max_row = max(row_ids)

        match_pass, orig_hash, replay_hash = engine.replay_historical_interval(min_row, max_row)
        assert match_pass is True, f"Replay hash mismatch! original={orig_hash}, replay={replay_hash}"
        assert orig_hash == replay_hash

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
