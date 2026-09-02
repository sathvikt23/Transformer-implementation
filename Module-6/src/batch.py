import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from src.tokenizer import PAD_TOKEN_ID, EOD_TOKEN_ID
from src.packer import PackedSequence


@dataclass
class MicroBatch:
    microbatch_id: str
    input_ids: List[int]
    labels: List[int]
    loss_mask: List[float]
    attention_mask: List[List[int]]  # 2D sequence-length binary/causal attention mask matrix
    position_ids: List[int]
    loss_mask_hash: str
    batch_hash: str
    sample_ids: List[str]
    spans: List[Dict[str, Any]]
    policy: str
    utilization: float


class BatchBuilder:
    """
    Constructs PyTorch / NN-ready microbatches from PackedSequence objects.
    Enforces invariant: loss mask NEVER covers padding tokens.
    Generates document-isolated causal attention masks to block cross-document leakage.
    """

    def __init__(self, seq_len: int = 512):
        self.seq_len = seq_len

    def build_microbatch(self, microbatch_id: str, packed_seq: PackedSequence) -> MicroBatch:
        input_ids = packed_seq.input_ids
        
        # Labels for Causal LM (target token is next token, with -100 for pad tokens)
        labels = []
        loss_mask = []
        position_ids = []

        # Enforce invariant: Loss mask NEVER covers padding (PAD_TOKEN_ID)
        for idx, tok in enumerate(input_ids):
            if tok == PAD_TOKEN_ID:
                labels.append(-100)
                loss_mask.append(0.0)
            else:
                labels.append(tok)
                loss_mask.append(1.0)

        # Build position IDs resetting per document span
        pos_id = 0
        curr_span_idx = 0
        spans = packed_seq.spans

        for idx, tok in enumerate(input_ids):
            if tok == PAD_TOKEN_ID:
                position_ids.append(0)
                continue

            # Check if entering new document span
            if curr_span_idx < len(spans):
                sp = spans[curr_span_idx]
                if idx >= sp["end_idx"]:
                    curr_span_idx += 1
                    pos_id = 0
            position_ids.append(pos_id)
            pos_id += 1

        # Build 2D Causal Attention Mask with Cross-Document Isolation
        attn_matrix = [[0 for _ in range(self.seq_len)] for _ in range(self.seq_len)]
        
        for i in range(self.seq_len):
            if input_ids[i] == PAD_TOKEN_ID:
                continue

            # Find document index for position i
            doc_i = None
            for sp in spans:
                if sp["start_idx"] <= i < sp["end_idx"]:
                    doc_i = sp["doc_id"]
                    break

            for j in range(i + 1):  # Causal lower-triangular
                if input_ids[j] == PAD_TOKEN_ID:
                    continue

                if packed_seq.policy == "structure-preserving":
                    # Strictly forbid cross-document attention leakage
                    doc_j = None
                    for sp in spans:
                        if sp["start_idx"] <= j < sp["end_idx"]:
                            doc_j = sp["doc_id"]
                            break
                    if doc_i is not None and doc_i == doc_j:
                        attn_matrix[i][j] = 1
                else:
                    attn_matrix[i][j] = 1

        # Compute deterministic hashes
        loss_mask_bytes = json.dumps(loss_mask).encode("utf-8")
        loss_mask_hash = hashlib.sha256(loss_mask_bytes).hexdigest()

        batch_content = {
            "microbatch_id": microbatch_id,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "spans": spans,
        }
        batch_hash = hashlib.sha256(json.dumps(batch_content, sort_keys=True).encode("utf-8")).hexdigest()

        return MicroBatch(
            microbatch_id=microbatch_id,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attn_matrix,
            position_ids=position_ids,
            loss_mask_hash=loss_mask_hash,
            batch_hash=batch_hash,
            sample_ids=packed_seq.sample_ids,
            spans=spans,
            policy=packed_seq.policy,
            utilization=packed_seq.utilization,
        )
