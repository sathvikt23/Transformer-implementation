import os
import json
from typing import Dict, Any, Tuple
from src.shards import Shard
from src.eval_firewall import EvalFirewall


def build_manifest_dict(shard: Shard, tokenizer_hash: str) -> Dict[str, Any]:
    """
    Constructs the canonical JSON-serializable manifest dictionary for a Shard.
    """
    return {
        "shard_id": shard.shard_id,
        "source_ids": [doc.doc_id for doc in shard.documents],
        "document_ids": [doc.doc_id for doc in shard.documents],
        "tokenizer_hash": tokenizer_hash,
        "token_count": shard.token_count,
        "language_script": shard.language_script,
        "capability_lane": shard.capability_lane,
        "license_provenance_tier": shard.license_provenance_tier,
        "license_tier": shard.license_provenance_tier,
        "cleaning_pipeline_hash": shard.cleaning_pipeline_hash,
        "dedup_status": shard.dedup_status,
        "pii_screen_status": "screened",
        "contamination_status": shard.contamination_status,
        "eval_overlap_status": "clear" if not shard.eval_overlap_status else "overlap",
        "content_hash": shard.content_hash,
        "parent_shard_ids": shard.parent_shard_ids,
        "parent_manifest_ids": shard.parent_shard_ids,
        "admission": "Pending validation",
    }


def validate_shard_manifest(
    manifest: Dict[str, Any],
    expected_tokenizer_hash: str,
    eval_firewall: EvalFirewall,
) -> Tuple[bool, str]:
    """
    Admission gate validating shard manifests before inclusion in training mixtures.
    Returns (admitted: bool, reason: str).
    """
    # 1. Tokenizer Integrity Check
    manifest_tok_hash = manifest.get("tokenizer_hash")
    if manifest_tok_hash != expected_tokenizer_hash:
        manifest["admission"] = "Blocked: tokenizer mismatch"
        return False, f"tokenizer_hash_mismatch (expected {expected_tokenizer_hash[:8]}, got {manifest_tok_hash[:8] if manifest_tok_hash else 'None'})"

    # 2. Evaluation Firewall Check
    blocked, reason = eval_firewall.is_blocked(manifest)
    if blocked:
        manifest["admission"] = f"Blocked: {reason}"
        return False, reason

    # 3. Contamination / Dedup Status Check
    if manifest.get("contamination_status") != "clean":
        manifest["admission"] = "Blocked: contaminated"
        return False, "contaminated"
    if "passed" not in manifest.get("dedup_status", ""):
        manifest["admission"] = "Blocked: dedup failed"
        return False, "dedup_failed"

    manifest["admission"] = "Admitted to registry"
    return True, "admitted"


def save_manifest(manifest: Dict[str, Any], output_dir: str) -> str:
    """
    Saves manifest dictionary as formatted JSON in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{manifest['shard_id']}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return filepath
