import os
import json
import sqlite3
from typing import Dict, Any, List, Optional


class ConsumptionLedger:
    """
    Append-only Consumption Ledger backed by SQLite in WAL mode.
    Records served microbatches, token spans, loss mask hashes, and checkpoint pointers.
    STRICT BOUNDARY: Does NOT import OpenTelemetry.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS microbatch_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                branch_id TEXT NOT NULL,
                global_step INTEGER NOT NULL,
                checkpoint_id TEXT,
                rank INTEGER NOT NULL,
                microbatch_id TEXT NOT NULL,
                packed_sample_ids TEXT NOT NULL,
                shard_ids TEXT NOT NULL,
                token_span_ids TEXT NOT NULL,
                loss_mask_hash TEXT NOT NULL,
                mixture_lane TEXT NOT NULL,
                curriculum_stage TEXT NOT NULL,
                tokenizer_version TEXT NOT NULL,
                dataloader_version TEXT NOT NULL,
                opus_decision_id TEXT,
                batch_hash TEXT NOT NULL,
                trace_id TEXT,
                span_id TEXT
            );
            """)
            conn.commit()

    def record_microbatch(
        self,
        run_id: str,
        branch_id: str,
        global_step: int,
        checkpoint_id: Optional[str],
        rank: int,
        microbatch_id: str,
        packed_sample_ids: List[str],
        shard_ids: List[str],
        token_span_ids: List[Dict[str, Any]],
        loss_mask_hash: str,
        mixture_lane: str,
        curriculum_stage: str,
        tokenizer_version: str,
        dataloader_version: str,
        opus_decision_id: Optional[str],
        batch_hash: str,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
    ) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO microbatch_ledger (
                run_id, branch_id, global_step, checkpoint_id, rank, microbatch_id,
                packed_sample_ids, shard_ids, token_span_ids, loss_mask_hash,
                mixture_lane, curriculum_stage, tokenizer_version, dataloader_version,
                opus_decision_id, batch_hash, trace_id, span_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                branch_id,
                global_step,
                checkpoint_id,
                rank,
                microbatch_id,
                json.dumps(packed_sample_ids),
                json.dumps(shard_ids),
                json.dumps(token_span_ids),
                loss_mask_hash,
                mixture_lane,
                curriculum_stage,
                tokenizer_version,
                dataloader_version,
                opus_decision_id,
                batch_hash,
                trace_id,
                span_id,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_max_ledger_offset(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(id) FROM microbatch_ledger;")
            row = cursor.fetchone()
            return row[0] if row[0] is not None else 0

    def get_microbatch_at_step(self, global_step: int, branch_id: str = "br_main") -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM microbatch_ledger
            WHERE global_step = ? AND branch_id = ?
            ORDER BY id DESC LIMIT 1;
            """, (global_step, branch_id))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["packed_sample_ids"] = json.loads(d["packed_sample_ids"])
                d["shard_ids"] = json.loads(d["shard_ids"])
                d["token_span_ids"] = json.loads(d["token_span_ids"])
                return d
            return None

    def fetch_range(self, from_offset: int, to_offset: int) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM microbatch_ledger
            WHERE id >= ? AND id <= ?
            ORDER BY id ASC;
            """, (from_offset, to_offset))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["packed_sample_ids"] = json.loads(d["packed_sample_ids"])
                d["shard_ids"] = json.loads(d["shard_ids"])
                d["token_span_ids"] = json.loads(d["token_span_ids"])
                results.append(d)
            return results

    def export_to_jsonl(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM microbatch_ledger ORDER BY id ASC;")
            rows = cursor.fetchall()
            with open(filepath, "w", encoding="utf-8") as f:
                for r in rows:
                    d = dict(r)
                    for k in ("packed_sample_ids", "shard_ids", "token_span_ids"):
                        if k in d and isinstance(d[k], str):
                            try:
                                d[k] = json.loads(d[k])
                            except Exception:
                                pass
                    f.write(json.dumps(d) + "\n")


class LearningLedger:
    """
    Learning Ledger backed by SQLite in WAL mode.
    Tracks token loss, perplexity clusters, loss deltas, and gradient norms per sample/shard.
    STRICT BOUNDARY: Does NOT import OpenTelemetry.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS learning_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                global_step INTEGER NOT NULL,
                shard_id TEXT NOT NULL,
                sample_id TEXT NOT NULL,
                lane TEXT NOT NULL,
                avg_token_loss REAL NOT NULL,
                high_ppl_clusters TEXT NOT NULL,
                loss_delta REAL NOT NULL,
                gradient_norm REAL NOT NULL,
                opus_score REAL NOT NULL,
                model_phase TEXT NOT NULL,
                usefulness_classification TEXT NOT NULL
            );
            """)
            conn.commit()

    def record_learning_event(
        self,
        global_step: int,
        shard_id: str,
        sample_id: str,
        lane: str,
        avg_token_loss: float,
        high_ppl_clusters: List[int],
        loss_delta: float,
        gradient_norm: float,
        opus_score: float,
        model_phase: str,
        usefulness_classification: str,
    ):
        with self._get_connection() as conn:
            conn.execute("""
            INSERT INTO learning_ledger (
                global_step, shard_id, sample_id, lane, avg_token_loss,
                high_ppl_clusters, loss_delta, gradient_norm, opus_score,
                model_phase, usefulness_classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                global_step,
                shard_id,
                sample_id,
                lane,
                avg_token_loss,
                json.dumps(high_ppl_clusters),
                loss_delta,
                gradient_norm,
                opus_score,
                model_phase,
                usefulness_classification,
            ))
            conn.commit()

    def export_to_jsonl(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM learning_ledger ORDER BY id ASC;")
            rows = cursor.fetchall()
            with open(filepath, "w", encoding="utf-8") as f:
                for r in rows:
                    d = dict(r)
                    if "high_ppl_clusters" in d and isinstance(d["high_ppl_clusters"], str):
                        try:
                            d["high_ppl_clusters"] = json.loads(d["high_ppl_clusters"])
                        except Exception:
                            pass
                    f.write(json.dumps(d) + "\n")

