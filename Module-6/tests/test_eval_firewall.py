from src.tokenizer import get_global_tokenizer
from src.shards import Shard, Document
from src.manifest import build_manifest_dict, validate_shard_manifest
from src.eval_firewall import EvalFirewall


def test_eval_firewall_blocks_tainted_shard():
    tokenizer = get_global_tokenizer()
    firewall = EvalFirewall()

    # Clean shard
    clean_doc = Document(doc_id="clean_doc_01", text="Clean training text", tokens=tokenizer.encode("Clean training text"))
    clean_shard = Shard.create(
        shard_id="clean_shard_001",
        documents=[clean_doc],
        language_script="en-US",
        capability_lane="general",
        license_provenance_tier="permissive-v1",
        cleaning_pipeline_hash="clean_hash",
        dedup_status="passed",
        contamination_status="clean",
        eval_overlap_status=False,
    )
    clean_m = build_manifest_dict(clean_shard, tokenizer.hash)
    admitted, reason = validate_shard_manifest(clean_m, tokenizer.hash, firewall)
    assert admitted is True, f"Clean shard should be admitted, got reason: {reason}"

    # Tainted shard with eval_overlap_status = True
    eval_doc = Document(doc_id="eval_doc_01", text="Evaluation benchmark text", tokens=tokenizer.encode("Evaluation benchmark text"))
    eval_shard = Shard.create(
        shard_id="eval_003",
        documents=[eval_doc],
        language_script="en-US",
        capability_lane="reasoning",
        license_provenance_tier="eval-holdout",
        cleaning_pipeline_hash="clean_hash",
        dedup_status="passed",
        contamination_status="clean",
        eval_overlap_status=True,
    )
    eval_m = build_manifest_dict(eval_shard, tokenizer.hash)
    admitted, reason = validate_shard_manifest(eval_m, tokenizer.hash, firewall)
    assert admitted is False, "Tainted eval shard must be blocked by EvalFirewall"
    assert reason in ("never_train_flag", "registered_eval_shard_id")
