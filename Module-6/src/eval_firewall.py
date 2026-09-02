from typing import Set, Dict, Any, Tuple


class EvalFirewall:
    """
    Eval/Validation Firewall enforcing never-train rules.
    Prevents contamination by blocking evaluation holdout shards from entering training pipelines.
    """

    def __init__(self):
        self.blocked_dataset_ids: Set[str] = {"eval_benchmark_v1", "gsm8k_eval", "mmlu_eval", "eval_003"}
        self.blocked_document_ids: Set[str] = set()
        self.audit_log: list = []

    def register_eval_dataset(self, dataset_id: str):
        self.blocked_dataset_ids.add(dataset_id)

    def register_eval_document(self, doc_id: str):
        self.blocked_document_ids.add(doc_id)

    def is_blocked(self, shard_data: Any) -> Tuple[bool, str]:
        """
        Evaluates whether a shard or manifest dictionary must be blocked from training.
        Returns (is_blocked, reason).
        """
        if isinstance(shard_data, dict):
            shard_id = shard_data.get("shard_id", "")
            eval_overlap = shard_data.get("eval_overlap_status", False)
            never_train = shard_data.get("never_train_flag", False)
            docs = shard_data.get("documents", [])
        else:
            shard_id = getattr(shard_data, "shard_id", "")
            eval_overlap = getattr(shard_data, "eval_overlap_status", False)
            never_train = getattr(shard_data, "never_train_flag", False) if hasattr(shard_data, "never_train_flag") else False
            docs = getattr(shard_data, "documents", [])

        if never_train or eval_overlap is True or eval_overlap in ("overlap", "tainted", "flagged"):
            reason = "never_train_flag"
            self.audit_log.append({"shard_id": shard_id, "status": "blocked", "reason": reason})
            return True, reason

        if shard_id in self.blocked_dataset_ids:
            reason = "registered_eval_shard_id"
            self.audit_log.append({"shard_id": shard_id, "status": "blocked", "reason": reason})
            return True, reason

        for doc in docs:
            doc_id = doc.get("doc_id") if isinstance(doc, dict) else getattr(doc, "doc_id", "")
            if doc_id in self.blocked_document_ids:
                reason = "registered_eval_doc_id"
                self.audit_log.append({"shard_id": shard_id, "status": "blocked", "reason": reason, "doc_id": doc_id})
                return True, reason

        self.audit_log.append({"shard_id": shard_id, "status": "admitted", "reason": "clean"})
        return False, ""

    def export_firewall_json(self, filepath: str):
        import json
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "firewall_rules": {
                "blocked_dataset_ids": sorted(list(self.blocked_dataset_ids)),
                "blocked_document_ids": sorted(list(self.blocked_document_ids)),
            },
            "audit_log": self.audit_log,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

