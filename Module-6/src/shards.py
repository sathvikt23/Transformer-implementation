import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from src.tokenizer import FrozenTokenizer, get_global_tokenizer


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    tokens: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Shard:
    shard_id: str
    documents: List[Document]
    token_count: int
    language_script: str
    capability_lane: str
    license_provenance_tier: str
    cleaning_pipeline_hash: str
    dedup_status: str
    contamination_status: str
    eval_overlap_status: bool
    content_hash: str
    parent_shard_ids: List[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        shard_id: str,
        documents: List[Document],
        language_script: str,
        capability_lane: str,
        license_provenance_tier: str,
        cleaning_pipeline_hash: str,
        dedup_status: str,
        contamination_status: str,
        eval_overlap_status: bool,
        parent_shard_ids: List[str] = None,
    ) -> "Shard":
        if parent_shard_ids is None:
            parent_shard_ids = []

        total_tokens = sum(len(doc.tokens) for doc in documents)

        # Compute content-addressed hash deterministically
        hasher = hashlib.sha256()
        hasher.update(shard_id.encode("utf-8"))
        hasher.update(capability_lane.encode("utf-8"))
        hasher.update(str(eval_overlap_status).encode("utf-8"))
        for doc in documents:
            hasher.update(doc.doc_id.encode("utf-8"))
            for tok in doc.tokens:
                hasher.update(tok.to_bytes(4, byteorder="big"))
        content_hash = hasher.hexdigest()

        return cls(
            shard_id=shard_id,
            documents=documents,
            token_count=total_tokens,
            language_script=language_script,
            capability_lane=capability_lane,
            license_provenance_tier=license_provenance_tier,
            cleaning_pipeline_hash=cleaning_pipeline_hash,
            dedup_status=dedup_status,
            contamination_status=contamination_status,
            eval_overlap_status=eval_overlap_status,
            content_hash=content_hash,
            parent_shard_ids=parent_shard_ids,
        )


def create_synthetic_shards(tokenizer: FrozenTokenizer = None, count: int = 24) -> List[Shard]:
    """
    Creates a set of deterministic synthetic shards across capability lanes.
    """
    if tokenizer is None:
        tokenizer = get_global_tokenizer()

    lanes = [
        ("code", "code-python", "commercial"),
        ("reasoning", "en-US", "permissive-v1"),
        ("indic", "hi-IN", "permissive-v1"),
        ("general", "en-US", "permissive-v1"),
    ]

    shards = []
    cleaning_hash = hashlib.sha256(b"v5_cleaning_v1").hexdigest()[:16]

    for i in range(1, count + 1):
        lane_idx = (i - 1) % len(lanes)
        lane, lang, tier = lanes[lane_idx]
        shard_id = f"shard_{i:04d}"

        # Build 5 documents per shard
        docs = []
        for d in range(1, 6):
            doc_id = f"doc_{shard_id}_{d:02d}"
            sample_text = f"Synthetic training text for {shard_id} document {d} in lane {lane}. Standard content with detailed logic patterns."
            tokens = tokenizer.encode(sample_text, add_eod=True)
            docs.append(Document(doc_id=doc_id, text=sample_text, tokens=tokens, metadata={"domain": lane}))

        shard = Shard.create(
            shard_id=shard_id,
            documents=docs,
            language_script=lang,
            capability_lane=lane,
            license_provenance_tier=tier,
            cleaning_pipeline_hash=cleaning_hash,
            dedup_status="passed_exact_and_near_dedup",
            contamination_status="clean",
            eval_overlap_status=False,
            parent_shard_ids=[],
        )
        shards.append(shard)

    return shards
